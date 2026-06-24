import os, re, sys, time, types, joblib, traceback
from pathlib import Path

import numpy as np
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from dotenv import load_dotenv

# ── Load .env reliably, regardless of cwd ───────────────────────────────────
# Looks next to this file first, then walks up — fixes the "key is blank
# because streamlit was launched from a different folder" problem.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH if _ENV_PATH.exists() else None)
load_dotenv()  # fallback: also check cwd / default search

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ── New Gemini SDK (old google-generativeai is EOL as of Nov 30 2025) ──────
try:
    from google import genai as genai_sdk
    from google.genai import types as genai_types
except ImportError:
    genai_sdk = None
    genai_types = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

try:
    from rank_bm25 import BM25Okapi
    BM25_SHIM_ACTIVE = False
except ImportError:
    BM25_SHIM_ACTIVE = True
    class BM25Okapi:
        def __init__(self, corpus=None, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
            self.corpus_size=0; self.avgdl=0.0; self.doc_freqs=[]; self.idf={}
            self.doc_len=[]; self.k1=k1; self.b=b
        def get_scores(self, query):
            if not self.corpus_size: return np.array([])
            score=np.zeros(self.corpus_size); dl=np.array(self.doc_len); avg=self.avgdl or 1.0
            for t in query:
                tf=np.array([(d.get(t) or 0) for d in self.doc_freqs])
                score+=(self.idf.get(t) or 0.0)*(tf*(self.k1+1)/(tf+self.k1*(1-self.b+self.b*dl/avg)))
            return score
    mod=types.ModuleType("rank_bm25"); mod.BM25Okapi=BM25Okapi; sys.modules["rank_bm25"]=mod

try:
    import gdown
except ImportError:
    gdown = None

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "").strip()
# gemini-3.5-flash is the current fast model as of mid-2026. Override via .env
# if you need a different tier (e.g. GEMINI_MODEL=gemini-3.1-flash-lite).
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()

# All 4 required files — chatbot_model.pkl downloaded from Drive if missing
REQUIRED = [
    "chatbot_model.pkl",
    "chatbot_tfidf.pkl",
    "chatbot_w2v.pkl",
    "chatbot_data.pkl",
]

# Google Drive file IDs — make each file public (Anyone with link → Viewer)
DRIVE_FILES = {
    "chatbot_model.pkl" : "17GettyBRuqyOOhnIykDztpfNdXr3zhp9",
    "chatbot_tfidf.pkl" : "1pOtcIAew-2NDMoGFb3wUTFcAQ1drtNYS",
    "chatbot_w2v.pkl"   : "1rtW9vgEKAVWDUYjTcaxxOwI89Exr5X8Y",
    "chatbot_data.pkl"  : "1ImK1-3uWekAZ_Km3tr2cIUeLzH9bk5_y",
}

STOP_WORDS = set(stopwords.words("english"))

CHAPTERS = {
    "Z":"General Information & Foreword", "A":"General Conditions of Service",
    "B":"Conduct & Discipline",           "C":"Pay & Allowances",
    "D":"TA/DA Rules",                    "E":"Medical Rules",
    "F":"Leave Rules",                    "G":"House Building Advance",
    "H":"Vehicle Advance",                "I":"Multi-Purpose Advance",
    "J":"Recruitment Rules",              "K":"Housing Allotment",
    "L":"Staff Welfare",                  "M":"Post Retirement",
    "N":"Leave Travel Concession",
}

DIRECT = [
    (["managing director","md name","who is md","current md","vikas kumar","md of dmrc"],
     "The Managing Director of DMRC is **Dr. Vikas Kumar**.\n_(Source: HR Compendium 2025, signed 4th August 2025, New Delhi)_"),
    (["compendium date","when published","published date"],
     "The HR Compendium 2025 was published on **4th August 2025**, signed by Dr. Vikas Kumar, MD, DMRC."),
]

QUESTIONS = [
    "Who is the Managing Director of DMRC?","How many days of casual leave does an employee get?",
    "What is the maternity leave duration?","Rules for earned leave encashment",
    "What is the House Building Advance limit?","Interest rate on house building advance",
    "Eligibility criteria for HBA","Medical attendance rules for retired employees",
    "What allowances are paid to employees?","Dearness Allowance calculation",
    "Rules for travelling allowance on tour","Daily allowance rates","Vehicle advance eligibility",
    "Multi purpose advance amount","Recruitment process in DMRC","Housing allotment rules",
    "Staff welfare fund usage","Post retirement contractual engagement",
    "Leave Travel Concession entitlement","LTC for home town","Conduct rules for employees",
    "Disciplinary action procedure","Appointment and confirmation rules",
    "Performance appraisal process","Hours of work and holidays",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def _download_from_drive(filename, file_id):
    """Download a file from Google Drive using gdown."""
    if gdown is None:
        st.error("gdown is not installed. Add `gdown` to requirements.txt and redeploy.")
        st.stop()
    url = f"https://drive.google.com/uc?id={file_id}"
    st.info(f"Downloading {filename} from Google Drive…")
    try:
        gdown.download(url, filename, quiet=False)
    except Exception as e:
        st.error(f"Download failed for {filename}: {e}")
        st.stop()
    if not os.path.exists(filename):
        st.error(f"Download completed but {filename} still not found.")
        st.stop()

def tokenize(text):
    try:    tokens = word_tokenize(text.lower())
    except: tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [t for t in tokens if t.isalpha()]

def build_context(results):
    out = []
    for i, (chunk, ch, score) in enumerate(results[:4], 1):
        page = ""
        m = re.search(rf'\b{re.escape(ch)}-(\d+)\b', str(chunk))
        if m: page = f"\nPage ref: {ch}-{m.group(1)}"
        out.append(f"Source {i}\nChapter: {ch} - {CHAPTERS.get(ch,'')} {page}"
                   f"\nSimilarity: {score:.2f}\nExcerpt: {' '.join(str(chunk).split())}")
    return "\n\n".join(out)

def _gemini_client():
    """Lazily build a genai client. Returns (client, error_message)."""
    if genai_sdk is None:
        return None, ("google-genai package is not installed. Run:\n"
                       "  pip install google-genai --break-system-packages")
    if not GEMINI_KEY:
        return None, (f"GEMINI_API_KEY is empty. Checked: {_ENV_PATH} "
                       f"(exists={_ENV_PATH.exists()}) and current working directory. "
                       "Make sure your .env file sits next to app.py and contains "
                       "GEMINI_API_KEY=your_key_here with no quotes.")
    try:
        client = genai_sdk.Client(api_key=GEMINI_KEY)
        return client, None
    except Exception as e:
        return None, f"Failed to create Gemini client: {e}"

def ask_gemini(query, results):
    """
    Returns (answer_text, error_text_or_None).
    error_text is shown to the user AND kept so st.rerun() doesn't hide it.
    """
    client, err = _gemini_client()
    if client is None:
        if not results:
            return "No Gemini API key configured.", err
        fallback = "\n\n".join(f"**Chapter {ch}**: {c[:300]}..." for c, ch, _ in results[:2])
        return fallback, err

    prompt = (
        f"You are the DMRC HR assistant. Answer using HR manual excerpts below.\n\n"
        f"Question:\n{query}\n\nHR manual excerpts:\n{build_context(results)}"
        if results else
        f"You are the DMRC HR assistant. Answer clearly.\n\nQuestion:\n{query}"
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        ans = (getattr(response, "text", "") or "").strip()
        if ans:
            return ans, None

        # No exception, but also no text — surface *why* instead of guessing.
        finish_reason = None
        try:
            finish_reason = response.candidates[0].finish_reason
        except Exception:
            pass
        err = f"Gemini returned an empty response (finish_reason={finish_reason})."
    except Exception as e:
        err = f"Gemini error [{type(e).__name__}]: {e}"

    if results:
        fallback = "\n\n".join(f"**Chapter {ch}**: {c[:300]}..." for c, ch, _ in results[:2])
        return fallback, err
    return "Could not generate answer.", err

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_data():
    for filename, file_id in DRIVE_FILES.items():
        if not os.path.exists(filename):
            _download_from_drive(filename, file_id)

    missing = [f for f in REQUIRED if not os.path.exists(f)]
    if missing:
        st.error(f"Missing files: {missing}\nPlace them next to app.py or add their Drive IDs to DRIVE_FILES.")
        st.stop()

    import torch
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "map_location": "cpu"})
    try:
        return {
            "st":    joblib.load("chatbot_model.pkl"),
            "tfidf": joblib.load("chatbot_tfidf.pkl"),
            "w2v":   joblib.load("chatbot_w2v.pkl"),
            "data":  joblib.load("chatbot_data.pkl"),
        }
    except Exception as e:
        st.error(f"Load error: {e}")
        st.stop()
    finally:
        torch.load = _orig

@st.cache_resource(show_spinner=False)
def load_ce():
    if CrossEncoder is None: return None
    try:    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except: return None

# ── Retrieval ─────────────────────────────────────────────────────────────────
def _st_scores(query, m):
    embs  = m["data"]["st_embeddings"].astype(np.float32)
    qv    = m["st"].encode([query], convert_to_numpy=True, normalize_embeddings=True)
    norms = np.linalg.norm(embs, axis=1, keepdims=True); norms[norms==0] = 1
    return (qv @ (embs/norms).T)[0]

def _bm25_scores(query, m):
    bm25 = m["data"].get("bm25")
    if bm25 is None: return None
    toks = [t for t in tokenize(query) if t not in STOP_WORDS]
    if not toks: return np.zeros(len(m["data"]["all_chunks"]))
    s = bm25.get_scores(toks); mx = s.max() if s.max() > 0 else 1.0
    return s / mx

def search(query, m, k=4, ce=None):
    q = query.lower()
    for kws, ans in DIRECT:
        if any(kw in q for kw in kws): return ans, None, None, [], None

    chunks, meta = m["data"]["all_chunks"], m["data"]["chunk_meta"]
    st_s   = _st_scores(query, m)
    bm25_s = _bm25_scores(query, m)
    combined = 0.55*st_s + 0.45*bm25_s if bm25_s is not None else st_s
    pool     = np.argsort(combined)[::-1][:k*4]
    cands    = [(chunks[i], meta[i], float(combined[i])) for i in pool]

    if ce:
        try:
            sc    = ce.predict([(query, c[0]) for c in cands])
            cands = [x[0] for x in sorted(zip(cands, sc), key=lambda x: x[1], reverse=True)[:k]]
        except: cands = cands[:k]
    else: cands = cands[:k]

    score     = cands[0][2] if cands else 0
    threshold = float(m["data"].get("similarity_threshold", 0.15))
    use_pdf   = cands and score >= threshold
    answer, gemini_err = ask_gemini(query, cands if use_pdf else [])
    chaps     = list(dict.fromkeys(c[1] for c in cands))[:3] if use_pdf else None
    return answer, score, chaps, cands, gemini_err

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DMRC HR Chatbot", page_icon="🚇", layout="centered")

models = load_data()
ce     = load_ce()

with st.sidebar:
    st.title("🚇 DMRC HR Bot")
    st.markdown("---")
    st.markdown("**Sample Questions**")
    st.caption("Click to ask instantly")
    for q in QUESTIONS:
        if st.button(q, key=f"sq_{q}"):
            st.session_state["pending"] = q
    st.markdown("---")
    if st.button("🗑 Clear chat"):
        st.session_state.history = []; st.rerun()
    if BM25_SHIM_ACTIVE:
        st.caption("ℹ️ BM25 shim active (rank_bm25 not installed)")

    # Diagnostics — shows immediately whether the key/SDK/model setup is sane,
    # without needing to send a chat message first.
    with st.expander("🔧 Gemini diagnostics"):
        st.write("**.env path checked:**", str(_ENV_PATH))
        st.write("**.env exists:**", _ENV_PATH.exists())
        st.write("**API key loaded:**", "Yes" if GEMINI_KEY else "❌ No / empty")
        st.write("**Model:**", GEMINI_MODEL)
        st.write("**google-genai installed:**", genai_sdk is not None)
        if st.button("Run test call"):
            client, err = _gemini_client()
            if err:
                st.error(err)
            else:
                try:
                    test = client.models.generate_content(model=GEMINI_MODEL, contents="Say OK")
                    st.success(f"Success: {getattr(test, 'text', test)!r}")
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")
                    st.code(traceback.format_exc())

if models is None:
    st.error("Missing .pkl files. Place all 4 next to app.py and restart."); st.stop()

if "history" not in st.session_state: st.session_state.history = []
if "pending" not in st.session_state: st.session_state.pending = None

st.title("DMRC HR Assistant")
st.caption("Ask questions about DMRC HR policies, leave, pay, advances, and more.")

# History entries are (role, msg, error_or_None) so warnings survive st.rerun().
for entry in st.session_state.history:
    role, msg = entry[0], entry[1]
    err = entry[2] if len(entry) > 2 else None
    with st.chat_message(role):
        st.write(msg)
        if err:
            st.warning(err)

query = st.chat_input("Example: How many days of casual leave am I entitled to?")

if st.session_state.pending and not query:
    query = st.session_state.pending
    st.session_state.pending = None

if query:
    st.session_state.pending = None
    st.chat_message("user").write(query)
    with st.chat_message("assistant"):
        with st.spinner("Searching HR Compendium..."):
            t0 = time.time()
            answer, score, chaps, cands, gemini_err = search(query, models, k=4, ce=ce)
            elapsed = round(time.time()-t0, 2)
        st.write(answer)
        if gemini_err:
            st.warning(gemini_err)
        if chaps:
            labels = []
            for c in chaps:
                page_ref = ""
                for chunk, ch2, _ in cands:
                    if ch2 == c:
                        pm = re.search(rf'\b{re.escape(c)}-(\d+)\b', str(chunk))
                        if pm: page_ref = f" (Page {c}-{pm.group(1)})"; break
                labels.append(f"Chapter {c}: {CHAPTERS.get(c,'')}{page_ref}")
            st.caption("📖 Source: " + "  |  ".join(labels) + f"  •  {elapsed}s")
        elif score is None:
            st.caption("📖 Source: HR Compendium 2025 — Foreword")
        else:
            st.caption(f"💬 General knowledge  •  {elapsed}s")
    st.session_state.history.append(("user", query, None))
    st.session_state.history.append(("assistant", answer, gemini_err))
    st.rerun()
