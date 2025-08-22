# app_part1.py — Part 1/3 of app.py
# Save parts 1..3 and concatenate in order to produce the runnable app.py:
#   cat app_part1.py app_part2.py app_part3.py > app.py

import streamlit as st
import os
import io
import sys
import json
import re
import hashlib
from typing import List, Any, TypedDict, Dict, Tuple
import PyPDF2
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import asyncio
import nest_asyncio
from dotenv import load_dotenv
import numpy as np
import time

# additional imports for persistence / auth / incremental training
import sqlite3
import secrets
import base64
from datetime import datetime, timedelta

# Optional heavy backends (graceful fallback)
try:
    import torch  # noqa: F401
    from colpali_engine.models import ColPali, ColQwen, ColSmol
    from pdf2image import convert_from_bytes
    COLPALI_AVAILABLE = True
except Exception:
    COLPALI_AVAILABLE = False

try:
    import pytesseract  # noqa: F401
    PYTESS_AVAILABLE = True
except Exception:
    PYTESS_AVAILABLE = False

# sklearn / joblib availability detection
try:
    from sklearn.linear_model import SGDClassifier
    import joblib
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    joblib = None

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False

load_dotenv()

# ---------- basic secrets helpers ----------
def _get_secret(name: str):
    val = os.environ.get(name)
    if val:
        return val
    try:
        if hasattr(st, "secrets") and st.secrets.get(name):
            return st.secrets.get(name)
    except Exception:
        pass
    return None

GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")
TAVILY_API_KEY = _get_secret("TAVILY_API_KEY")

# ---------- misc helpers ----------
def _short_hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]

# Safe async runner for Streamlit
def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

# ---------- State type (extended) ----------
class State(TypedDict):
    documents: List[Dict[str, Any]]
    question: str
    llm: Any
    embeddings_model: Any
    full_document_text: str
    text_chunks: List[Dict[str, Any]]
    vector_store: Any
    retrieved_docs: List[Dict[str, Any]]
    confidence: float
    web_results: List[str]
    answer: str
    file_hash: str
    user_id: int

# ---------- persistence + multi-user configuration ----------
DB_PATH = os.environ.get("AGENTIC_RAG_DB_PATH", "/tmp/agentic_rag.db")
STORAGE_ROOT = os.environ.get("AGENTIC_RAG_STORAGE", "/tmp/agentic_rag_storage")
os.makedirs(STORAGE_ROOT, exist_ok=True)

def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE,
                    password_hash TEXT,
                    salt TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    ts REAL,
                    role TEXT,
                    content TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    question TEXT,
                    snippet TEXT,
                    label INTEGER,
                    ts REAL
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    filename TEXT,
                    file_hash TEXT,
                    uploaded_at REAL
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS rate_limits (
                    user_id INTEGER PRIMARY KEY,
                    window_start REAL,
                    count INTEGER
                )''')
    conn.commit()
    return conn

_db_conn = init_db()

# ---------- auth helpers ----------
def _hash_password(password: str, salt: str = None):
    if salt is None:
        salt = secrets.token_hex(16)
    pwd = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
    return pwd.hex(), salt

def create_user(username: str, password: str):
    c = _db_conn.cursor()
    pwd_hash, salt = _hash_password(password)
    try:
        c.execute("INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)", (username, pwd_hash, salt))
        _db_conn.commit()
        return True, None
    except sqlite3.IntegrityError:
        return False, "username already exists"

def verify_user(username: str, password: str):
    c = _db_conn.cursor()
    c.execute("SELECT id, password_hash, salt FROM users WHERE username=?", (username,))
    row = c.fetchone()
    if not row:
        return None
    uid, stored_hash, salt = row
    pwd_hash, _ = _hash_password(password, salt)
    if secrets.compare_digest(pwd_hash, stored_hash):
        return uid
    return None

# ---------- rate limiter (simple, per-minute) ----------
RATE_LIMIT_N = int(os.environ.get("AGENTIC_RAG_RATE_LIMIT_N", "30"))

def check_rate_limit(user_id: int):
    now = time.time()
    c = _db_conn.cursor()
    c.execute("SELECT window_start, count FROM rate_limits WHERE user_id=?", (user_id,))
    row = c.fetchone()
    if not row:
        c.execute("INSERT OR REPLACE INTO rate_limits (user_id, window_start, count) VALUES (?, ?, ?)",
                  (user_id, now, 1))
        _db_conn.commit()
        return True
    window_start, count = row
    if now - window_start > 60:
        c.execute("UPDATE rate_limits SET window_start=?, count=? WHERE user_id=?", (now, 1, user_id))
        _db_conn.commit()
        return True
    if count >= RATE_LIMIT_N:
        return False
    c.execute("UPDATE rate_limits SET count=? WHERE user_id=?", (count + 1, user_id))
    _db_conn.commit()
    return True

# ---------- file sanitization and saving ----------
def sanitize_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)

MAX_UPLOAD_BYTES = int(os.environ.get("AGENTIC_RAG_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))  # 10 MB

def save_uploaded_file(user_id: int, uploaded_file):
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("File too large")
    safe_name = sanitize_filename(uploaded_file.name)
    file_hash = _short_hash_bytes(raw)
    user_folder = os.path.join(STORAGE_ROOT, f"user_{user_id}")
    os.makedirs(user_folder, exist_ok=True)
    path = os.path.join(user_folder, f"{file_hash}_{safe_name}")
    with open(path, "wb") as fh:
        fh.write(raw)
    c = _db_conn.cursor()
    c.execute("INSERT INTO files (user_id, filename, file_hash, uploaded_at) VALUES (?, ?, ?, ?)",
              (user_id, safe_name, file_hash, time.time()))
    _db_conn.commit()
    return {"path": path, "file_hash": file_hash, "filename": safe_name, "raw": raw}

# ---------- model file helpers (for reranker management) ----------
def get_user_model_dir(user_id: int) -> str:
    user_model_dir = os.path.join(STORAGE_ROOT, f"user_{user_id}", "models")
    os.makedirs(user_model_dir, exist_ok=True)
    return user_model_dir

def list_user_models(user_id: int) -> List[str]:
    d = get_user_model_dir(user_id)
    try:
        files = sorted([f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))])
        return files
    except Exception:
        return []

def save_uploaded_model_file(user_id: int, uploaded_file) -> str:
    """Save uploaded model file (.joblib, .pkl) into user's models dir and return saved path."""
    data = uploaded_file.getvalue()
    fname = sanitize_filename(uploaded_file.name)
    d = get_user_model_dir(user_id)
    path = os.path.join(d, fname)
    with open(path, "wb") as fh:
        fh.write(data)
    return path

def load_user_reranker(user_id: int, fname: str = None) -> Tuple[bool, str]:
    """Attempt to load a user reranker model. Returns (loaded_bool, path_or_error)."""
    if not SKLEARN_AVAILABLE or joblib is None:
        return False, "scikit-learn / joblib not available in environment"
    d = get_user_model_dir(user_id)
    if fname is None:
        path = os.path.join(d, "reranker_sgd.joblib")
        if not os.path.exists(path):
            files = list_user_models(user_id)
            path = os.path.join(d, files[0]) if files else None
    else:
        path = os.path.join(d, fname)
    if not path or not os.path.exists(path):
        return False, "no reranker model found for user"
    try:
        clf = joblib.load(path)
        st.session_state["reranker_model"] = clf
        st.session_state["loaded_reranker"] = path
        return True, path
    except Exception as e:
        return False, f"failed to load model: {e}"

# ---------- embedding cache helper ----------
def embed_texts_wrap(emb_model, texts):
    try:
        return np.array(emb_model.embed_documents(texts))
    except Exception:
        try:
            return np.array(emb_model.embed(texts))
        except Exception:
            try:
                return np.array([emb_model.encode(t) for t in texts])
            except Exception:
                return np.zeros((len(texts), 768))

def embed_texts_cached(emb_model, texts, user_id, file_hash):
    cache_dir = os.path.join(STORAGE_ROOT, f"user_{user_id}", "emb_cache", file_hash)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "embeddings.npy")
    try:
        if os.path.exists(cache_file):
            cached = np.load(cache_file)
            if cached.shape[0] == len(texts):
                return cached
        X = embed_texts_wrap(emb_model, texts)
        np.save(cache_file, X)
        return X
    except Exception:
        return embed_texts_wrap(emb_model, texts)

# ---------- ColPali helpers (optional + crash-proof) ----------

def colpali_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

class ColpaliIndex:
    def __init__(self, model_name: str, user_id: int, file_hash: str):
        self.model_name = model_name
        self.user_id = user_id
        self.file_hash = file_hash
        self.device = colpali_device()
        self.model = None
        self.cache_dir = os.path.join(STORAGE_ROOT, f"user_{user_id}", "colpali", file_hash)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_model(self):
        if self.model is not None:
            return
        if not COLPALI_AVAILABLE:
            return
        try:
            import torch
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            if self.model_name.lower().endswith("qwen"):
                self.model = ColQwen.from_pretrained(self.model_name, torch_dtype=dtype).to(self.device)
            elif self.model_name.lower().endswith("smol") or self.model_name.lower().endswith("smo"):
                self.model = ColSmol.from_pretrained(self.model_name, torch_dtype=dtype).to(self.device)
            else:
                self.model = ColPali.from_pretrained(self.model_name, torch_dtype=dtype).to(self.device)
        except Exception:
            self.model = None

    def _page_key(self, doc_id: int, page: int) -> str:
        return f"d{doc_id}_p{page}"

    def _emb_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.npy")

    def ensure_page_embeddings(self, docs: List[Dict[str, Any]]):
        if not COLPALI_AVAILABLE:
            return False
        self._load_model()
        if self.model is None:
            return False
        try:
            for i, d in enumerate(docs):
                pages = convert_from_bytes(d["file_content"], dpi=110, fmt="png")
                for p_idx, img in enumerate(pages, start=1):
                    key = self._page_key(i, p_idx)
                    out_path = self._emb_path(key)
                    if os.path.exists(out_path):
                        continue
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    try:
                        emb = self.model.encode_images([buf.read()])
                        np.save(out_path, emb.detach().cpu().numpy())
                    except Exception:
                        continue
            return True
        except Exception:
            return False

    def query_score(self, query: str, doc_id: int, page: int) -> float:
        if not COLPALI_AVAILABLE or self.model is None:
            return 0.0
        try:
            key = self._page_key(doc_id, page)
            path = self._emb_path(key)
            if not os.path.exists(path):
                return 0.0
            img_emb = np.load(path)
            try:
                q_emb = self.model.encode_queries([query])
                q = q_emb.detach().cpu().numpy()[0]
            except Exception:
                return 0.0
            v = img_emb[0]
            num = float((q * v).sum())
            den = float(np.linalg.norm(q) * np.linalg.norm(v) + 1e-8)
            return num / den if den > 0 else 0.0
        except Exception:
            return 0.0

# End of part 1


# app_part2.py — Part 2/3 of app.py
# Continue from part1. This contains the RAG workflow, chunking, retrieval, reranking, scoring and generation nodes.


def run_agentic_rag():
    web_search_tool = TavilySearchResults(max_results=4, tavily_api_key=TAVILY_API_KEY)

    def initialize_models_node(state: State) -> Dict[str, Any]:
        if state.get("llm") and state.get("embeddings_model"):
            return {}
        if "llm" in st.session_state and "embeddings_model" in st.session_state:
            return {"llm": st.session_state["llm"], "embeddings_model": st.session_state["embeddings_model"]}
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        return {"llm": llm, "embeddings_model": embeddings_model}

    def load_and_chunk_docs_node(state: State) -> Dict[str, Any]:
        out_chunks = []
        docs = state.get("documents", []) or []
        for i, doc in enumerate(docs):
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(doc["file_content"]))
                for p, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    # Optional OCR for image-only pages
                    if not page_text.strip() and PYTESS_AVAILABLE:
                        try:
                            from pdf2image import convert_from_bytes
                            imgs = convert_from_bytes(doc["file_content"], dpi=110, first_page=p+1, last_page=p+1)
                            if imgs:
                                page_text = pytesseract.image_to_string(imgs[0]) or ""
                        except Exception:
                            pass
                    out_chunks.append({
                        "text": page_text or "",
                        "doc_id": i,
                        "filename": doc.get("filename", f"doc_{i}"),
                        "page": p + 1
                    })
            except Exception:
                out_chunks.append({
                    "text": "",
                    "doc_id": i,
                    "filename": doc.get("filename", f"doc_{i}"),
                    "page": 0
                })
        max_tokens = 1000
        overlap = 200
        combined = []
        if TIKTOKEN_AVAILABLE:
            try:
                enc = tiktoken.encoding_for_model("gemini-1.5-flash")
                for c in out_chunks:
                    tokens = enc.encode(c["text"])
                    i2 = 0
                    while i2 < len(tokens):
                        chunk_tokens = tokens[i2:i2 + max_tokens]
                        chunk_text = enc.decode(chunk_tokens)
                        combined.append({"text": chunk_text, "doc_id": c["doc_id"], "filename": c["filename"], "page": c["page"]})
                        i2 += max_tokens - overlap
                return {"text_chunks": combined}
            except Exception:
                pass
        chunk_size = 2000
        overlap_chars = 400
        for c in out_chunks:
            t = c["text"]
            if t is None:
                t = ""
            if not t:
                combined.append({"text": "", "doc_id": c["doc_id"], "filename": c["filename"], "page": c["page"]})
                continue
            for start in range(0, len(t), chunk_size - overlap_chars):
                combined.append({
                    "text": t[start:start + chunk_size],
                    "doc_id": c["doc_id"],
                    "filename": c["filename"],
                    "page": c["page"]
                })
        return {"text_chunks": combined}

    def create_chroma_index_node(state: State) -> Dict[str, Any]:
        try:
            file_hash = state.get("file_hash")
            user_id = state.get("user_id", "anon")
            cache_key = f"chroma_{user_id}_{file_hash}" if file_hash else None
            if cache_key and cache_key in st.session_state:
                return {"vector_store": st.session_state[cache_key]}
            chunks = state.get("text_chunks", []) or []
            if not chunks or not state.get("embeddings_model"):
                return {"vector_store": None}
            texts = [c["text"] for c in chunks]
            metadatas = [{"doc_id": c["doc_id"], "filename": c["filename"], "page": c["page"], "chunk_index": idx} for idx, c in enumerate(chunks)]
            persist_dir = os.path.join(STORAGE_ROOT, f"user_{user_id}", "chroma", file_hash or "nohash")
            os.makedirs(persist_dir, exist_ok=True)
            try:
                vector_store = Chroma.from_texts(texts=texts, embedding=state["embeddings_model"], metadatas=metadatas, persist_directory=persist_dir)
                if hasattr(vector_store, "persist"):
                    try:
                        vector_store.persist()
                    except Exception:
                        pass
            except TypeError:
                vector_store = Chroma.from_texts(texts=texts, embedding=state["embeddings_model"], metadatas=metadatas)
            if cache_key:
                st.session_state[cache_key] = vector_store
            st.session_state["chunk_store"] = chunks
            return {"vector_store": vector_store}
        except Exception as e:
            print(f"create_chroma_index_node error: {e}")
            return {"vector_store": None}

    def retrieve_node(state: State) -> Dict[str, Any]:
        if state.get("vector_store") is None:
            return {"retrieved_docs": []}
        try:
            retriever = state["vector_store"].as_retriever(search_kwargs={"k": 6})
            docs = retriever.invoke(state["question"]) if hasattr(retriever, "invoke") else retriever.get_relevant_documents(state["question"]) 
            out = []
            for d in docs:
                content = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
                meta = getattr(d, "metadata", {}) if hasattr(d, "metadata") else {}
                out.append({"text": content, "metadata": meta})
            return {"retrieved_docs": out}
        except Exception as e:
            print(f"retrieve_node error: {e}")
            return {"retrieved_docs": []}

    def colpali_hybrid_score(snippets: List[Dict[str, Any]], question: str, user_id: int, file_hash: str) -> List[Dict[str, Any]]:
        cfg = st.session_state.get("colpali_cfg")
        if not cfg or not cfg.get("enabled"):
            return snippets
        index: ColpaliIndex = cfg.get("index")
        if not index:
            return snippets
        rescored = []
        for s in snippets:
            meta = s.get("metadata", {})
            doc_id = int(meta.get("doc_id", 0))
            page = int(meta.get("page", 1))
            try:
                img_sim = index.query_score(question, doc_id, page)
            except Exception:
                img_sim = 0.0
            prev = float(s.get("score", 0.0))
            blended = 0.7 * img_sim + 0.3 * prev
            s2 = dict(s)
            s2["score"] = blended
            s2["colpali_sim"] = img_sim
            rescored.append(s2)
        rescored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return rescored

    def rerank_node(state: State) -> Dict[str, Any]:
        snippets = state.get("retrieved_docs", []) or []
        if not snippets:
            return {"retrieved_docs": []}
        clf = st.session_state.get("reranker_model")
        emb_model = state.get("embeddings_model")
        if clf is not None and SKLEARN_AVAILABLE:
            texts = [s.get("text", "") for s in snippets]
            user_id = state.get("user_id", "anon")
            file_hash = state.get("file_hash", "global")
            X = embed_texts_cached(emb_model, texts, user_id, file_hash + "_rerank")
            try:
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(np.array(X))[:, 1]
                elif hasattr(clf, "decision_function"):
                    scores = clf.decision_function(np.array(X))
                    probs = 1 / (1 + np.exp(-scores))
                elif hasattr(clf, "predict"):
                    preds = clf.predict(np.array(X))
                    probs = np.array(preds, dtype=float)
                else:
                    probs = np.zeros((len(X),))
                for i, s in enumerate(snippets):
                    s["score"] = float(probs[i])
            except Exception:
                pass
        try:
            user_id = state.get("user_id", "anon")
            file_hash = state.get("file_hash", "global")
            snippets = colpali_hybrid_score(snippets, state.get("question", ""), user_id, file_hash)
        except Exception:
            pass
        snippets.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return {"retrieved_docs": snippets}

    def score_snippets_node(state: State) -> Dict[str, Any]:
        snippets = state.get("retrieved_docs", []) or []
        llm = state.get("llm")
        if not snippets or not llm:
            return {"retrieved_docs": snippets}
        scored = []
        prompt = PromptTemplate(
            template="""You are an evaluator. For the given question and snippet return a JSON object {\"confidence\": <float 0.0-1.0>} estimating how well the snippet answers the question. Respond ONLY with the JSON.

Question: {question}
Snippet:
{snippet}""",
            input_variables=["question", "snippet"],
        )
        for s in snippets:
            snippet_text = s.get("text", "")
            try:
                evaluator_chain = prompt | llm | JsonOutputParser()
                resp = evaluator_chain.invoke({"question": state["question"], "snippet": snippet_text})
                conf = float(resp.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            s_copy = dict(s)
            s_copy["llm_conf"] = conf
            if "score" not in s_copy:
                s_copy["score"] = conf
            scored.append(s_copy)
        scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return {"retrieved_docs": scored}

    def evaluate_node(state: State) -> Dict[str, Any]:
        prompt = PromptTemplate(
            template="""You are a grader assessing the relevance of retrieved snippets to a user question.
Aggregate the relevance of all snippets and provide a single confidence score from 0.0 to 1.0.
Respond ONLY with a JSON object containing a single key "confidence".

Question: {question}
Snippets:
{snippets}""",
            input_variables=["question", "snippets"],
        )
        evaluator_chain = prompt | state["llm"] | JsonOutputParser()
        snippets_text = "\n\n".join([s.get("text", "") for s in state.get("retrieved_docs", [])])
        confidence = 0.0
        try:
            response_json = evaluator_chain.invoke({"question": state["question"], "snippets": snippets_text})
            confidence = float(response_json.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        return {"confidence": confidence}

    def web_search_node(state: State) -> Dict[str, Any]:
        hits = []
        try:
            results = web_search_tool.invoke({"query": state["question"]})
            hits = [r.get("content", "") for r in results if isinstance(r, dict) and r.get("content")]
        except Exception:
            hits = []
        return {"web_results": hits}

    def stream_answer(llm, prompt, placeholder):
        if hasattr(llm, "stream") or hasattr(llm, "astream") or hasattr(llm, "invoke_stream"):
            try:
                stream = None
                if hasattr(llm, "invoke_stream"):
                    stream = llm.invoke_stream(prompt)
                elif hasattr(llm, "stream"):
                    stream = llm.stream(prompt)
                elif hasattr(llm, "astream"):
                    stream = llm.astream(prompt)
                buf = ""
                for part in stream:
                    piece = getattr(part, "content", None) or str(part)
                    buf += piece
                    placeholder.markdown(buf)
                return buf
            except Exception as e:
                print(f"streaming error: {e}")
        out = None
        try:
            out = llm.invoke(prompt)
            ans = getattr(out, "content", str(out))
        except Exception:
            ans = str(out) if out is not None else ""
        sentences = re.split(r'(?<=[.!?])\s+', ans)
        buf = ""
        for s in sentences:
            buf += s + " "
            placeholder.markdown(buf)
            time.sleep(0.12)
        return buf

    def generate_node(state: State) -> Dict[str, Any]:
        retrieved = state.get("retrieved_docs", []) or []
        context = ""
        if retrieved and isinstance(retrieved, list):
            top = retrieved[0]
            context = top.get("text", "")
        elif state.get("web_results"):
            context = "\n\n".join(state.get("web_results", []))
        else:
            return {"answer": "I couldn't find any relevant information in the PDFs or online."}

        prompt = (
            f"You are a helpful assistant. Answer the question using only the context below. "
            f"If no answer can be found, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state['question']}\n\nAnswer:"
        )
        try:
            placeholder_key = f"stream_placeholder_{state.get('file_hash','')}_{hash(state['question'])}"
            placeholder = st.session_state.get(placeholder_key)
            if placeholder is None:
                placeholder = st.empty()
                st.session_state[placeholder_key] = placeholder
            ans = stream_answer(state["llm"], prompt, placeholder)
            return {"answer": ans}
        except Exception as e:
            print(f"generate_node error: {e}")
            try:
                response = state["llm"].invoke(prompt)
                ans = getattr(response, "content", str(response))
                return {"answer": ans}
            except Exception as e2:
                print(f"generate_node fallback error: {e2}")
                return {"answer": "An error occurred while generating the answer."}

    def summarize_node(state: State) -> Dict[str, Any]:
        doc_text = ""
        for d in state.get("documents", []):
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(d["file_content"]))
                for p in reader.pages:
                    page_text = p.extract_text() or ""
                    doc_text += page_text + "\n\n"
            except Exception:
                pass
        prompt = f"Summarize the following documents. Keep the summary concise but complete, include section headings if helpful.\n\n{doc_text[:15000]}"
        try:
            response = state["llm"].invoke(prompt)
            ans = getattr(response, "content", str(response))
            return {"answer": ans}
        except Exception as e:
            print(f"summarize_node error: {e}")
            return {"answer": "An error occurred while summarizing the document."}

    # Build workflow
    workflow = StateGraph(State)
    workflow.add_node("initialize_models", initialize_models_node)
    workflow.add_node("load_and_chunk_docs", load_and_chunk_docs_node)
    workflow.add_node("create_chroma_index", create_chroma_index_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("score_snippets", score_snippets_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("summarize", summarize_node)

    workflow.set_entry_point("initialize_models")
    workflow.add_edge("initialize_models", "load_and_chunk_docs")
    workflow.add_edge("load_and_chunk_docs", "create_chroma_index")
    workflow.add_edge("create_chroma_index", "retrieve")

    def route_after_retrieval(state: State):
        return "summarize" if state["question"].lower().strip() == "summarize" else "rerank"

    def route_after_evaluation(state: State):
        return "generate" if state.get("confidence", 0.0) >= 0.6 else "web_search"

    workflow.add_conditional_edges("retrieve", route_after_retrieval, {"summarize": "summarize", "rerank": "rerank"})
    workflow.add_edge("rerank", "score_snippets")
    workflow.add_edge("score_snippets", "evaluate")
    workflow.add_conditional_edges("evaluate", route_after_evaluation, {"generate": "generate", "web_search": "web_search"})
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("summarize", END)
    return workflow.compile()

# ---------- small async wrapper ----------
async def get_rag_response(rag_app, initial_state):
    return await rag_app.ainvoke(initial_state)

# ---------- keep original train_reranker_from_feedback API ----------
def train_reranker_from_feedback():
    current_user = st.session_state.get("user_id")
    if current_user is None:
        st.warning("Login first")
        return
    train_reranker_incremental(current_user)

# End of part 2


# app_part3.py — Part 3/3 of app.py
# Final UI + main entrypoint. This file completes the app when concatenated after part1 and part2.

# ---------- Reranker incremental training (continued) ----------

def train_reranker_incremental(user_id:int):
    if not SKLEARN_AVAILABLE:
        st.warning("scikit-learn not available in the environment")
        return
    c = _db_conn.cursor()
    c.execute("SELECT question, snippet, label FROM feedback WHERE user_id=?", (user_id,))
    rows = c.fetchall()
    if not rows:
        st.warning("No feedback examples to train on")
        return
    texts = [r[1] for r in rows]
    labels = [r[2] for r in rows]
    emb_model = st.session_state.get("embeddings_model")
    file_hash = st.session_state.get("last_file_hash", "global")
    X = embed_texts_cached(emb_model, texts, user_id, f"{file_hash}_feedback")
    user_model_dir = get_user_model_dir(user_id)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    model_path_ts = os.path.join(user_model_dir, f"reranker_sgd_{timestamp}.joblib")
    model_path_stable = os.path.join(user_model_dir, "reranker_sgd.joblib")
    clf = None
    if joblib and os.path.exists(model_path_stable):
        try:
            clf = joblib.load(model_path_stable)
        except Exception:
            clf = None
    try:
        if clf is None:
            clf = SGDClassifier(loss="log_loss", max_iter=5)
            clf.partial_fit(X, labels, classes=np.array([0,1]))
        else:
            clf.partial_fit(X, labels)
        if joblib:
            joblib.dump(clf, model_path_ts)
            joblib.dump(clf, model_path_stable)
            st.success(f"Reranker trained and saved to {model_path_ts}")
            st.session_state["reranker_model"] = clf
            st.session_state["last_trained_reranker"] = model_path_ts
            st.session_state["loaded_reranker"] = model_path_stable
        else:
            st.warning("joblib not available; trained model not saved to disk")
    except Exception as e:
        st.error(f"Failed to train reranker incrementally: {e}")

# ---------- small helpers (persist/load UI) ----------

def persist_chat(user_id:int, role:str, content:str):
    try:
        c = _db_conn.cursor()
        c.execute("INSERT INTO chats (user_id, ts, role, content) VALUES (?, ?, ?, ?)",
                  (user_id, time.time(), role, content))
        _db_conn.commit()
    except Exception:
        pass

    try:
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        if role == "user":
            st.session_state["chat_history"].append({"user": content, "assistant": ""})
        elif role == "assistant":
            if st.session_state["chat_history"] and st.session_state["chat_history"][-1].get("assistant", "") == "":
                st.session_state["chat_history"][-1]["assistant"] = content
            else:
                st.session_state["chat_history"].append({"user": "", "assistant": content})
    except Exception:
        pass


def load_user_chats(user_id: int, limit: int = 500):
    try:
        cur = _db_conn.cursor()
        cur.execute("SELECT role, content, ts FROM chats WHERE user_id=? ORDER BY ts ASC LIMIT ?", (user_id, limit))
        rows = cur.fetchall()
        conv: List[Dict[str, str]] = []
        pending_user = None
        for role, content, ts in rows:
            if role == "user":
                pending_user = content
            elif role == "assistant":
                conv.append({"user": pending_user or "", "assistant": content})
                pending_user = None
        if pending_user is not None:
            conv.append({"user": pending_user, "assistant": ""})
        st.session_state["chat_history"] = conv
    except Exception:
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

# ---------- existing model helpers (ensure_models) ----------

def ensure_models():
    if "llm" not in st.session_state or "embeddings_model" not in st.session_state:
        if not GEMINI_API_KEY:
            st.error("GEMINI_API_KEY not found.")
            st.stop()
        st.session_state["llm"] = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)
        st.session_state["embeddings_model"] = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    return st.session_state["llm"], st.session_state["embeddings_model"]


def _short_hash_bytes_multiple(docs: List[Dict[str, Any]]) -> str:
    m = hashlib.sha256()
    for d in docs:
        m.update(d["file_content"])
    return m.hexdigest()[:12]

# ---------- main app UI ----------

def main():
    nest_asyncio.apply()
    st.set_page_config(page_title="Agentic RAG Chatbot", layout="wide")
    st.title("Agentic RAG — Multi-PDF RAG with Active Learning & Streaming")

    if not GEMINI_API_KEY or not TAVILY_API_KEY:
        st.error("API keys for Gemini or Tavily are not configured.")
        st.stop()

    # session defaults
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "feedback_examples" not in st.session_state:
        st.session_state.feedback_examples = []
    if "reranker_model" not in st.session_state:
        st.session_state.reranker_model = None
    if "storage_root" not in st.session_state:
        st.session_state["storage_root"] = STORAGE_ROOT
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None
    if "username" not in st.session_state:
        st.session_state["username"] = None

    if st.session_state.get("user_id") and not st.session_state.get("chat_history"):
        try:
            load_user_chats(st.session_state.get("user_id"))
        except Exception:
            pass

    sidebar = st.sidebar
    sidebar.header("Account")

    auth_mode = sidebar.selectbox("Mode", ["Login", "Register"]) 
    auth_username = sidebar.text_input("Username", key="auth_username")
    auth_password = sidebar.text_input("Password", type="password", key="auth_password")
    if sidebar.button("Confirm Auth"):
        if auth_mode == "Register":
            ok, err = create_user(auth_username.strip(), auth_password)
            if ok:
                st.success("User created. Please login.")
            else:
                st.error(f"Registration error: {err}")
        else:
            uid = verify_user(auth_username.strip(), auth_password)
            if uid:
                st.session_state["user_id"] = uid
                st.session_state["username"] = auth_username.strip()
                try:
                    load_user_chats(uid)
                except Exception:
                    pass
                if SKLEARN_AVAILABLE and joblib:
                    ok, msg = load_user_reranker(uid)
                    if ok:
                        st.session_state["last_trained_reranker"] = st.session_state.get("loaded_reranker")
                st.success("Logged in")
            else:
                st.error("Login failed")

    if st.session_state.get("user_id"):
        sidebar.markdown(f"**Signed in as:** {st.session_state.get("username")}")
        if sidebar.button("Logout"):
            st.session_state["user_id"] = None
            st.session_state["username"] = None
            st.success("Logged out")

    sidebar.header("Controls")
    uploaded_files = sidebar.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
    if sidebar.button("Clear Feedback"):
        st.session_state.feedback_examples = []
    save_path = sidebar.text_input("Model save dir", value="/tmp")
    st.session_state["reranker_path"] = save_path

    # Reranker management UI (upload / list / load / delete)
    sidebar.markdown("---")
    sidebar.subheader("Reranker model management")
    if SKLEARN_AVAILABLE and joblib:
        if st.session_state.get("user_id"):
            user_id = st.session_state.get("user_id")
            uploaded_model_file = sidebar.file_uploader("Upload reranker (.joblib/.pkl)", type=["joblib", "pkl"], key="upload_reranker")
            if uploaded_model_file is not None:
                try:
                    saved_path = save_uploaded_model_file(user_id, uploaded_model_file)
                    st.success(f"Saved model to {saved_path}")
                    ok, msg = load_user_reranker(user_id, os.path.basename(saved_path))
                    if ok:
                        st.success("Reranker uploaded and loaded into session.")
                    else:
                        st.warning(f"Saved but failed to load: {msg}")
                except Exception as e:
                    st.error(f"Failed to save uploaded model: {e}")

            models = list_user_models(user_id)
            selected_model = None
            if models:
                selected_model = sidebar.selectbox("Available rerankers", models, key="selected_reranker")
                if sidebar.button("Load reranker"):
                    ok, msg = load_user_reranker(user_id, selected_model)
                    if ok:
                        st.success(f"Loaded reranker: {msg}")
                    else:
                        st.error(f"Failed to load reranker: {msg}")
                if sidebar.button("Delete selected reranker"):
                    try:
                        os.remove(os.path.join(get_user_model_dir(user_id), selected_model))
                        st.success("Deleted model")
                    except Exception as e:
                        st.error(f"Failed to delete model: {e}")
            else:
                sidebar.info("No reranker models uploaded for this user yet.")
        else:
            sidebar.info("Login to manage reranker models")
    else:
        sidebar.info("scikit-learn / joblib not available in this environment. Model upload / load disabled.")

    # Manual feedback expander
    with sidebar.expander("Manual feedback (paste snippet + label)"):
        manual_question = st.text_input("Question (optional)", key="manual_feedback_question")
        manual_snippet = st.text_area("Snippet text", key="manual_feedback_snippet", height=150)
        manual_label = st.selectbox("Label", ["Relevant", "Not relevant"], key="manual_feedback_label")
        if st.button("Submit manual feedback"):
            current_user = st.session_state.get("user_id")
            if current_user is None:
                st.warning("Please login to submit feedback.")
            elif not manual_snippet.strip():
                st.warning("Paste a snippet to submit as feedback.")
            else:
                label_val = 1 if manual_label == "Relevant" else 0
                qtxt = manual_question.strip() or ""
                cur = _db_conn.cursor()
                cur.execute("INSERT INTO feedback (user_id, question, snippet, label, ts) VALUES (?, ?, ?, ?, ?)",
                          (current_user, qtxt, manual_snippet[:2000], label_val, time.time()))
                _db_conn.commit()
                st.success("Manual feedback saved.")
                st.session_state.setdefault("feedback_examples", []).append({
                    "user_id": current_user,
                    "question": qtxt,
                    "snippet": manual_snippet,
                    "label": label_val,
                    "ts": time.time(),
                })
                if SKLEARN_AVAILABLE:
                    try:
                        train_reranker_incremental(current_user)
                    except Exception as e:
                        st.warning(f"Retrainer failed: {e}")
                else:
                    st.info("Install scikit-learn to enable retrainer training")

    fb_rows = 0
    try:
        if st.session_state.get("user_id"):
            cur = _db_conn.cursor()
            cur.execute("SELECT COUNT(*) FROM feedback WHERE user_id=?", (st.session_state.get("user_id"),))
            fb_rows = cur.fetchone()[0]
    except Exception:
        fb_rows = len(st.session_state.get("feedback_examples", []))
    sidebar.markdown(f"**Feedback examples:** {fb_rows}")

    # ColPali hybrid controls
    sidebar.markdown("---")
    colpali_enabled = sidebar.checkbox("Enable ColPali/ColQwen hybrid rerank (experimental)", value=False)
    colpali_model = sidebar.selectbox("Col model", [
        "vidore/colpali-v1.3",
        "vidore/colqwen-v1.0",
        "vidore/colsmol-v1.0"
    ], index=0)
    if colpali_enabled and not COLPALI_AVAILABLE:
        sidebar.warning("ColPali backend not installed or torch/pdf2image unavailable. Falling back to text-only.")
        colpali_enabled = False

    if SKLEARN_AVAILABLE:
        if sidebar.button("Train reranker from feedback"):
            current_user = st.session_state.get("user_id")
            if current_user is None:
                st.warning("Please login to train your reranker.")
            else:
                train_reranker_incremental(current_user)
    else:
        sidebar.info("Install scikit-learn to enable reranker training")

    # handle uploaded files
    docs = []
    if uploaded_files:
        files_list = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
        saved_count = 0
        for f in files_list:
            try:
                if st.session_state.get("user_id"):
                    saved = save_uploaded_file(st.session_state["user_id"], f)
                    docs.append({"filename": saved["filename"], "file_content": saved["raw"]})
                else:
                    raw = f.getvalue()
                    if len(raw) > MAX_UPLOAD_BYTES:
                        st.warning(f"Skipped {f.name}: file too large for anonymous upload.")
                        continue
                    docs.append({"filename": f.name, "file_content": raw})
                saved_count += 1
            except Exception as e:
                st.error(f"Failed to save {f.name}: {e}")
        sidebar.success(f"{saved_count} file(s) processed")

    # Build ColPali page-embedding index (best-effort)
    if colpali_enabled and docs:
        try:
            current_user = st.session_state.get("user_id") or 0
            file_hash_preview = _short_hash_bytes_multiple(docs)
            cfg = {
                "enabled": True,
                "model_name": colpali_model,
                "index": None,
            }
            index = ColpaliIndex(colpali_model, current_user, file_hash_preview)
            ok = index.ensure_page_embeddings(docs)
            if ok:
                cfg["index"] = index
                st.session_state["colpali_cfg"] = cfg
                st.sidebar.success("ColPali page embeddings ready (cached)")
            else:
                st.session_state["colpali_cfg"] = {"enabled": False}
                st.sidebar.info("ColPali not used (prepare failed). Using text-only.")
        except Exception as e:
            st.session_state["colpali_cfg"] = {"enabled": False}
            st.sidebar.info(f"ColPali disabled: {e}")
    else:
        st.session_state["colpali_cfg"] = {"enabled": False}

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Chat History")
        if st.session_state.chat_history:
            for chat in reversed(st.session_state.chat_history[-200:]):
                st.markdown(f"**You:** {chat.get('user','')}")
                st.markdown(f"**Assistant:** {chat.get('assistant','')}")
                st.markdown("---")
        else:
            st.info("No messages yet")
        if st.button("Export chat JSON"):
            payload = json.dumps(st.session_state.chat_history, ensure_ascii=False, indent=2)
            st.download_button("Download chat JSON", payload, file_name="chat_history.json", mime="application/json")
        if st.button("Export feedback JSON"):
            payload = json.dumps(st.session_state.get("feedback_examples", []), ensure_ascii=False, indent=2)
            st.download_button("Download feedback JSON", payload, file_name="feedback.json", mime="application/json")
        if st.session_state.get("last_trained_reranker"):
            st.write("Last trained reranker:")
            st.write(st.session_state.get("last_trained_reranker"))
            if st.button("Download reranker model"):
                try:
                    with open(st.session_state["last_trained_reranker"], "rb") as fh:
                        st.download_button("Download model file", fh.read(), file_name=os.path.basename(st.session_state["last_trained_reranker"]))
                except Exception:
                    st.error("Failed to read saved model file")

    with col2:
        st.subheader("Ask across uploaded PDFs")
        user_query = st.text_input("Ask a question about the uploaded documents (or type 'summarize'):")
        if st.button("Get Answer"):
            if not user_query:
                st.warning("Please enter a question.")
            else:
                current_user = st.session_state.get("user_id")
                if current_user is None:
                    st.warning("Please login to use the app.")
                elif not check_rate_limit(current_user):
                    st.error("Rate limit exceeded. Try again later.")
                else:
                    with st.spinner("Thinking..."):
                        llm, embeddings_model = ensure_models()
                        if "rag_app" not in st.session_state:
                            st.session_state["rag_app"] = run_agentic_rag()
                        rag_app = st.session_state["rag_app"]
                        file_hash = _short_hash_bytes_multiple(docs) if docs else _short_hash_bytes(b"no-doc")
                        initial_state = {
                            "documents": docs,
                            "question": user_query,
                            "llm": llm,
                            "embeddings_model": embeddings_model,
                            "file_hash": file_hash,
                            "user_id": current_user
                        }
                        st.session_state["last_file_hash"] = file_hash
                        persist_chat(current_user, "user", user_query)
                        try:
                            final_state = run_async(get_rag_response(rag_app, initial_state))
                            answer = final_state.get("answer", "No answer was generated.")
                            retrieved = final_state.get("retrieved_docs", [])
                            confidence = final_state.get("confidence", 0.0)

                            st.markdown("## Answer")
                            st.write(answer)
                            st.markdown(f"**Confidence:** {confidence:.2f}")

                            st.markdown("### Best retrieved snippet (provenance shown). Label it to improve the reranker.")
                            if retrieved:
                                best = retrieved[0]
                                meta = best.get("metadata", {})
                                filename = meta.get("filename", "unknown")
                                page = meta.get("page", "?")
                                chunk_index = meta.get("chunk_index", 0)
                                score = best.get("score", None)
                                cps = best.get("colpali_sim", None)
                                st.markdown(f"**Best Source:** {filename} (p.{page}) chunk {chunk_index} — score: {score if score is not None else 'N/A'}  ")
                                if cps is not None:
                                    st.caption(f"ColPali page-image similarity: {cps:.3f}")
                                snippet_text = best.get("text", "")[:2500]
                                st.write(snippet_text)
                                qhash = abs(hash(user_query)) if user_query else 0
                                file_hash_short = file_hash or "nohash"
                                rel_key = f"mark_rel_{file_hash_short}_{chunk_index}_{qhash}"
                                notrel_key = f"mark_notrel_{file_hash_short}_{chunk_index}_{qhash}"
                                c1, c2 = st.columns(2)
                                if c1.button("Mark Relevant", key=rel_key):
                                    cur = _db_conn.cursor()
                                    cur.execute("INSERT INTO feedback (user_id, question, snippet, label, ts) VALUES (?, ?, ?, ?, ?)",
                                              (current_user, user_query, snippet_text[:2000], 1, time.time()))
                                    _db_conn.commit()
                                    st.success("Marked as relevant — feedback saved.")
                                    st.session_state.setdefault("feedback_examples", []).append({
                                        "user_id": current_user,
                                        "question": user_query,
                                        "snippet": snippet_text,
                                        "label": 1,
                                        "ts": time.time(),
                                    })
                                    if SKLEARN_AVAILABLE:
                                        try:
                                            train_reranker_incremental(current_user)
                                        except Exception as e:
                                            st.warning(f"Retrainer failed: {e}")
                                    else:
                                        st.info("Install scikit-learn to enable reranker training")
                                if c2.button("Mark Not Relevant", key=notrel_key):
                                    cur = _db_conn.cursor()
                                    cur.execute("INSERT INTO feedback (user_id, question, snippet, label, ts) VALUES (?, ?, ?, ?, ?)",
                                              (current_user, user_query, snippet_text[:2000], 0, time.time()))
                                    _db_conn.commit()
                                    st.success("Marked as not relevant — feedback saved.")
                                    st.session_state.setdefault("feedback_examples", []).append({
                                        "user_id": current_user,
                                        "question": user_query,
                                        "snippet": snippet_text,
                                        "label": 0,
                                        "ts": time.time(),
                                    })
                                    if SKLEARN_AVAILABLE:
                                        try:
                                            train_reranker_incremental(current_user)
                                        except Exception as e:
                                            st.warning(f"Retrainer failed: {e}")
                                    else:
                                        st.info("Install scikit-learn to enable reranker training")

                                if len(retrieved) > 1:
                                    st.markdown("---")
                                    st.markdown("### Other retrieved snippets — quick label")
                                    for idx, s in enumerate(retrieved[1:6], start=2):
                                        meta = s.get("metadata", {})
                                        fname = meta.get("filename", "unknown")
                                        pnum = meta.get("page", "?")
                                        chunk_idx = meta.get("chunk_index", 0)
                                        st.markdown(f"**#{idx}** Source: {fname} (p.{pnum}) chunk {chunk_idx}")
                                        txt = s.get("text", "")[:2000]
                                        st.write(txt)
                                        qh = abs(hash(user_query)) if user_query else 0
                                        rel_k = f"rel_{file_hash_short}_{chunk_idx}_{qh}"
                                        notrel_k = f"notrel_{file_hash_short}_{chunk_idx}_{qh}"
                                        b1, b2 = st.columns([1, 1])
                                        if b1.button("Relevant", key=rel_k):
                                            cur = _db_conn.cursor()
                                            cur.execute(
                                                "INSERT INTO feedback (user_id, question, snippet, label, ts) VALUES (?, ?, ?, ?, ?)",
                                                (current_user, user_query, txt, 1, time.time()),
                                            )
                                            _db_conn.commit()
                                            st.success("Saved relevant feedback.")
                                            st.session_state.setdefault("feedback_examples", []).append({
                                                "user_id": current_user,
                                                "question": user_query,
                                                "snippet": txt,
                                                "label": 1,
                                                "ts": time.time(),
                                            })
                                            if SKLEARN_AVAILABLE:
                                                try:
                                                    train_reranker_incremental(current_user)
                                                except Exception as e:
                                                    st.warning(f"Retrainer failed: {e}")
                                            else:
                                                st.info("Install scikit-learn to enable reranker training")
                                        if b2.button("Not relevant", key=notrel_k):
                                            cur = _db_conn.cursor()
                                            cur.execute(
                                                "INSERT INTO feedback (user_id, question, snippet, label, ts) VALUES (?, ?, ?, ?, ?)",
                                                (current_user, user_query, txt, 0, time.time()),
                                            )
                                            _db_conn.commit()
                                            st.success("Saved not-relevant feedback.")
                                            st.session_state.setdefault("feedback_examples", []).append({
                                                "user_id": current_user,
                                                "question": user_query,
                                                "snippet": txt,
                                                "label": 0,
                                                "ts": time.time(),
                                            })
                                            if SKLEARN_AVAILABLE:
                                                try:
                                                    train_reranker_incremental(current_user)
                                                except Exception as e:
                                                    st.warning(f"Retrainer failed: {e}")
                                            else:
                                                st.info("Install scikit-learn to enable reranker training")

                            else:
                                st.info("No retrieved snippets to display. Consider uploading PDFs or widening your search query.")

                            try:
                                persist_chat(current_user, "assistant", answer)
                            except Exception:
                                pass

                        except Exception as e:
                            st.error(f"Failed to get an answer: {e}")

        if st.button("Summarize documents"):
            current_user = st.session_state.get("user_id")
            if not docs:
                st.warning("Upload at least one PDF to summarize.")
            elif current_user is None:
                st.warning("Please login to use the app.")
            elif not check_rate_limit(current_user):
                st.error("Rate limit exceeded. Try again later.")
            else:
                with st.spinner("Summarizing documents..."):
                    llm, embeddings_model = ensure_models()
                    if "rag_app" not in st.session_state:
                        st.session_state["rag_app"] = run_agentic_rag()
                    rag_app = st.session_state["rag_app"]
                    file_hash = _short_hash_bytes_multiple(docs)
                    initial_state = {
                        "documents": docs,
                        "question": "summarize",
                        "llm": llm,
                        "embeddings_model": embeddings_model,
                        "file_hash": file_hash,
                        "user_id": current_user
                    }
                    st.session_state["last_file_hash"] = file_hash
                    persist_chat(current_user, "user", "summarize")
                    try:
                        final_state = run_async(get_rag_response(rag_app, initial_state))
                        answer = final_state.get("answer", "No summary generated.")
                        st.markdown("## Document Summary")
                        st.write(answer)
                        st.session_state["chat_history"].append({"user": "summarize", "assistant": answer})
                        persist_chat(current_user, "assistant", answer)
                    except Exception as e:
                        st.error(f"Error summarizing: {e}")

# If you want to run the app directly
if __name__ == "__main__":
    main()

# ---------------- Deployment notes ----------------
# 1. Create a GitHub repo and commit the concatenated app.py along with a requirements.txt.
# 2. Example requirements.txt (adjust versions as needed):
#    streamlit
#    PyPDF2
#    numpy
#    python-dotenv
#    langchain-google-genai (or the appropriate package name used in your environment)
#    langchain-core
#    langchain-community
#    tiktoken (optional)
#    scikit-learn (optional, for reranker)
#    joblib (optional)
#    pytesseract (optional)
#    pdf2image (optional)
