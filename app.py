import streamlit as st
import re
import fitz  # PyMuPDF
import docx
import uuid
import glob
import os
import hashlib
import pickle
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# ==================== Configuration ====================
st.set_page_config(page_title="MBE Document Assistant", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    .chunk-display {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        white-space: pre-wrap;
        font-family: 'Courier New', monospace;
        line-height: 1.6;
    }
    .answer-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Session State
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.files_data = {}
    st.session_state.collection = None
    st.session_state.messages = []  # للحفاظ على تاريخ الحوار

# Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
PDF_PASSWORD = "mbe2025"
DOCS_FOLDER = "/mount/src/test/documents"
CACHE_FOLDER = "./cache"

os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found!")

# ==================== Helper Functions ====================
def get_file_hash(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_cache(key):
    path = os.path.join(CACHE_FOLDER, f"{key}.pkl")
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None

def save_cache(key, data):
    path = os.path.join(CACHE_FOLDER, f"{key}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def format_table_as_structured_text(table, table_num=None):
    if not table:
        return ""
    headers = [clean_text(str(c)) for c in table[0]]
    text = f"\n📊 Table {table_num or ''}\n\n"
    if headers:
        text += "| " + " | ".join(headers) + " |\n"
        text += "| " + " --- |" * len(headers) + " |\n"
    for row in table[1:]:
        cells = [clean_text(str(c)) for c in row]
        if any(cells):
            text += "| " + " | ".join(cells) + " |\n"
    return text

def create_smart_chunks(text, size=1000, overlap=200):
    words = text.split()
    chunks = []
    if len(words) <= size:
        return [text] if text.strip() else []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if len(chunk.split()) >= 30:
            chunks.append(chunk)
    return chunks

# ==================== Extraction with Page Info ====================
def extract_pdf_detailed(filepath):
    try:
        doc = fitz.open(filepath)
        if doc.is_encrypted and not doc.authenticate(PDF_PASSWORD):
            return None, "Invalid password"
    except Exception as e:
        return None, str(e)

    file_info = {'chunks': [], 'total_pages': len(doc), 'total_tables': 0}

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = f"\n# Page {page_num + 1} - {os.path.basename(filepath)}\n\n"

        # Text
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b.get('type') == 0:
                text = ""
                for line in b.get('lines', []):
                    for span in line.get('spans', []):
                        text += span.get('text', '') + " "
                if text.strip():
                    page_text += clean_text(text) + "\n\n"

        # Tables
        tables = page.find_tables()
        for t_idx, t in enumerate(tables.tables, 1):
            file_info['total_tables'] += 1
            extracted = t.extract()
            if extracted:
                page_text += format_table_as_structured_text(extracted, file_info['total_tables']) + "\n\n"

        chunks = create_smart_chunks(page_text)
        file_info['chunks'].extend(chunks)

    doc.close()
    return file_info, None

# DOCX and TXT (simple)
def extract_docx_detailed(filepath):
    doc = docx.Document(filepath)
    text = "\n\n".join([clean_text(p.text) for p in doc.paragraphs if p.text.strip()])
    for i, t in enumerate(doc.tables, 1):
        rows = [[cell.text for cell in row.cells] for row in t.rows]
        text += format_table_as_structured_text(rows, i)
    chunks = create_smart_chunks(text)
    return {'chunks': chunks, 'total_pages': 1, 'total_tables': len(doc.tables)}, None

def extract_txt_detailed(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    chunks = create_smart_chunks(text)
    return {'chunks': chunks, 'total_pages': 1, 'total_tables': 0}, None

def get_files_from_folder():
    exts = ['*.pdf', '*.docx', '*.doc', '*.txt']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(DOCS_FOLDER, ext)))
    return files

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-large")

# ==================== Groq Answer with Citation ====================
def answer_question_with_groq(query, chunks, metadatas):
    if not GROQ_API_KEY:
        return "GROQ_API_KEY missing"

    context_parts = []
    for i, chunk in enumerate(chunks):
        meta = metadatas[i]
        page_match = re.search(r'# Page (\d+)', chunk)
        page = page_match.group(1) if page_match else "Unknown"
        context_parts.append(f"--- SOURCE {i+1} (File: {meta['source']}, Page: {page}) ---\n{chunk}\n--- END ---")

    context = "\n\n".join(context_parts)

    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": """You are a precise MBE document assistant at Hochschule Anhalt.
RULES:
1. Answer ONLY from sources.
2. ALWAYS cite: (File: X.pdf, Page: Y)
3. Count/list exactly from tables.
4. No info → "No sufficient information"
5. Use question language.
6. Concise."""},
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {query}\nAnswer with citations."}
        ],
        "temperature": 0.0,
        "max_tokens": 1500
    }

    try:
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                             headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, json=data, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

# ==================== Main App ====================
st.markdown("<div class='main-card'><h1 style='text-align: center;'>MBE Document Assistant</h1><p style='text-align: center;'>Ask about modules, thesis, exams, program rules</p></div>", unsafe_allow_html=True)

files = get_files_from_folder()
if not files:
    st.warning(f"No documents in {DOCS_FOLDER}")
else:
    st.success(f"Found {len(files)} documents")
    with st.expander("Available Files"):
        for f in files:
            st.write("• " + os.path.basename(f))

if files and st.button("🚀 Process All Documents", type="primary"):
    with st.spinner("Processing..."):
        files_data = {}
        all_chunks = []
        all_metadatas = []

        client = chromadb.Client()
        coll = client.create_collection(name=f"docs_{uuid.uuid4().hex[:8]}", embedding_function=get_embedding_function())

        for filepath in files:
            name = os.path.basename(filepath)
            ext = name.split('.')[-1].lower()
            hash_key = f"{get_file_hash(filepath)}_{ext}"
            cached = load_cache(hash_key)

            if cached:
                info = cached
            else:
                if ext == 'pdf':
                    info, err = extract_pdf_detailed(filepath)
                elif ext in ['docx', 'doc']:
                    info, err = extract_docx_detailed(filepath)
                else:
                    info, err = extract_txt_detailed(filepath)
                if err:
                    st.error(f"Error {name}: {err}")
                    continue
                save_cache(hash_key, info)

            files_data[name] = info
            for chunk in info['chunks']:
                all_chunks.append(chunk)
                page_match = re.search(r'# Page (\d+)', chunk)
                page = int(page_match.group(1)) if page_match else 1
                all_metadatas.append({"source": name, "page": page})

        if all_chunks:
            coll.add(documents=all_chunks, metadatas=all_metadatas, ids=[f"id_{i}" for i in range(len(all_chunks))])

        st.session_state.files_data = files_data
        st.session_state.collection = coll
        st.session_state.processed = True
        st.success("Processing completed!")
        st.balloons()

if st.session_state.processed:
    st.markdown("---")
    st.subheader("💬 Chat with MBE Documents")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                results = st.session_state.collection.query(query_texts=[query], n_results=10)
                answer = answer_question_with_groq(query, results["documents"][0], results["metadatas"][0])
                st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": answer})

        # Sources
        st.markdown("### 📄 Sources")
        for i, (chunk, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
            with st.expander(f"Source {i}: {meta['source']} - Page {meta['page']}"):
                st.markdown(f'<div class="chunk-display">{chunk}</div>', unsafe_allow_html=True)
else:
    st.info("Add your PDFs to /mount/src/test/documents and click Process")
