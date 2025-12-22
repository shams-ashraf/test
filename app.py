import streamlit as st
import re
import fitz  # PyMuPDF
import io
import docx
import uuid
import glob
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import requests
import json
from datetime import datetime
from io import BytesIO
import os
import time
import pickle
import hashlib

# Configuration
st.set_page_config(
    page_title="Smart Document Extractor",
    page_icon="📄",
    layout="wide"
)

# Custom CSS (مش هتغير)
st.markdown("""
<style>
    .main-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    .stat-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.files_data = {}
    st.session_state.collection = None

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
PDF_PASSWORD = os.getenv("PDF_PASSWORD", "mbe2025")  # أضفت الباسورد اللي قلت عليه
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "./documents")
CACHE_FOLDER = os.getenv("CACHE_FOLDER", "./cache")

os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found in environment variables!")

# ==================== Helper Functions (معدلة للدقة) ====================

def get_file_hash(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_cache(cache_key):
    cache_file = os.path.join(CACHE_FOLDER, f"{cache_key}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None

def save_cache(cache_key, data):
    cache_file = os.path.join(CACHE_FOLDER, f"{cache_key}.pkl")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        st.warning(f"⚠️ Could not save cache: {str(e)}")

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# جدول كـ Markdown (أهم تعديل – الـ LLM بيفهمها أحسن)
def format_table_as_structured_text(extracted_table, table_number=None, page_num=None):
    if not extracted_table or len(extracted_table) == 0:
        return ""
    
    headers = [str(cell).strip() if cell else f"Col_{i+1}" for i, cell in enumerate(extracted_table[0])]
    headers = [clean_text(h) for h in headers]
    
    md = f"\n## 📊 Table #{table_number} on Page {page_num}\n\n"
    md += "| " + " | ".join(headers) + " |\n"
    md += "| " + " --- |" * len(headers) + " |\n"
    
    row_count = 0
    for row in extracted_table[1:]:
        cells = [str(cell).strip() if cell else "" for cell in row]
        cells = [clean_text(c) for c in cells]
        if any(cells):
            md += "| " + " | ".join(cells) + " |\n"
            row_count += 1
    
    md += f"\n**Table Summary**: {row_count} data rows, {len(headers)} columns.\n"
    return md

def create_smart_chunks(text, chunk_size=1200, overlap=150):
    words = text.split()
    chunks = []
    if len(words) <= chunk_size:
        return [text] if text.strip() else []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        if len(chunk.split()) >= 50:
            chunks.append(chunk)
    return chunks

# ==================== PDF Extraction (معدلة بـ metadata دقيقة) ====================
def extract_pdf_detailed(filepath):
    try:
        doc = fitz.open(filepath)
        if doc.is_encrypted:
            if PDF_PASSWORD:
                if not doc.authenticate(PDF_PASSWORD):
                    doc.close()
                    return None, "❌ Invalid PDF password"
            else:
                doc.close()
                return None, "❌ PDF is password-protected"
    except Exception as e:
        return None, f"❌ Error opening PDF: {str(e)}"

    file_info = {
        'chunks': [],  # list of dicts: {'text': ..., 'metadata': {'source':..., 'page':..., 'table_num':...}}
        'total_pages': len(doc),
        'total_tables': 0,
    }

    global_table_counter = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = f"\n# Document Page {page_num + 1}\n\n"

        # Text
        text_blocks = page.get_text("dict")["blocks"]
        for block in text_blocks:
            if block.get('type') == 0:
                text_content = ""
                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        text_content += span.get('text', '') + " "
                if text_content.strip():
                    page_text += clean_text(text_content) + "\n\n"

        # Tables
        tables = page.find_tables()
        for _ in tables.tables:
            global_table_counter += 1
            file_info['total_tables'] += 1

        for table_idx, table in enumerate(tables.tables, 1):
            extracted = table.extract()
            if extracted:
                table_text = format_table_as_structured_text(
                    extracted, global_table_counter, page_num + 1
                )
                page_text += table_text + "\n\n"

        # Chunk the whole page (with metadata)
        page_chunks = create_smart_chunks(page_text, chunk_size=1200, overlap=150)
        for chunk in page_chunks:
            file_info['chunks'].append({
                'text': chunk,
                'metadata': {
                    'source': os.path.basename(filepath),
                    'page': page_num + 1,
                    'table_num': None  # يمكن تحسينه لو عايز تتبع كل جدول
                }
            })

    doc.close()
    return file_info, None

# DOCX و TXT (مش محتاجين تعديل كبير)
def extract_docx_detailed(filepath):
    doc = docx.Document(filepath)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(clean_text(para.text))
    for table in doc.tables:
        extracted = [[cell.text for cell in row.cells] for row in table.rows]
        full_text.append(format_table_as_structured_text(extracted, table_number="DOCX", page_num=1))
    
    text = "\n\n".join(full_text)
    chunks = create_smart_chunks(text, chunk_size=1200, overlap=150)
    return {
        'chunks': [{'text': c, 'metadata': {'source': os.path.basename(filepath), 'page': 1}} for c in chunks],
        'total_pages': 1,
        'total_tables': len(doc.tables)
    }, None

def extract_txt_detailed(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    chunks = create_smart_chunks(text, chunk_size=1200, overlap=150)
    return {
        'chunks': [{'text': c, 'metadata': {'source': os.path.basename(filepath), 'page': 1}} for c in chunks],
        'total_pages': 1,
        'total_tables': 0
    }, None

# ==================== Embedding & Collection ====================
def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large"
    )

# ==================== Groq Answer (أقوى prompt + metadata) ====================
def answer_question_with_groq(query, relevant_chunks, relevant_metadatas):
    if not GROQ_API_KEY:
        return "❌ GROQ_API_KEY missing"

    # Format context with citations
    context_parts = []
    for i, (chunk, meta) in enumerate(zip(relevant_chunks, relevant_metadatas), 1):
        context_parts.append(
            f"--- SOURCE {i} ---\n"
            f"File: {meta['source']} | Page: {meta['page']}\n"
            f"{chunk}\n"
            f"--- END SOURCE {i} ---"
        )
    context = "\n\n".join(context_parts)

    data = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are a PRECISE document Q&A assistant for Biomedical Engineering at Hochschule Anhalt.
RULES (STRICT):
1. Answer ONLY from the provided SOURCES below.
2. ALWAYS cite sources exactly: (File: X.pdf, Page: Y)
3. For counting (e.g., "how many modules"): List every matching item explicitly from tables.
4. NO guessing page numbers or details not in sources.
5. If conflicting info: Say "Sources show conflicting information".
6. If not found: "No information found in the documents."
7. Use the same language as the question (English/German/Arabic).
8. Be concise unless asked for details.
9. Tables are in Markdown format – read them carefully."""
            },
            {
                "role": "user",
                "content": f"Sources:\n{context}\n\nQuestion: {query}\n\nAnswer precisely with citations."
            }
        ],
        "temperature": 0.0,
        "max_tokens": 2000
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json=data,
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ Groq error: {str(e)}"

# ==================== Main App (معدل جزء الـ query فقط) ====================
# ... (كل الـ UI زي ما هو لحد الـ query part)

if st.session_state.processed:
    # ... (الإحصائيات والعرض زي ما هو)

    st.subheader("🔍 Ask about documents")
    query = st.text_input("Enter your question here...", key="query_input")

    col_search1, col_search2 = st.columns([3, 1])
    with col_search1:
        search_only = st.checkbox("Search only (no AI answer)", value=False)
    with col_search2:
        num_results = st.selectbox("Number of results", [5, 8, 12, 20], index=1)  # زدت الافتراضي لـ 8

    if query:
        with st.spinner("Searching..."):
            results = st.session_state.collection.query(
                query_texts=[query],
                n_results=num_results
            )

            if not search_only and GROQ_API_KEY:
                st.markdown("### 🤖 AI Answer:")
                with st.spinner("Generating precise answer..."):
                    answer = answer_question_with_groq(
                        query,
                        results["documents"][0],
                        results["metadatas"][0]  # مرر الـ metadata
                    )
                    st.markdown(f"""
<div class="answer-box">
    <h4 style='margin-top: 0;'>💡 Answer:</h4>
    {answer}
</div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📄 Source Chunks:")
            for idx, (chunk, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
                with st.expander(f"📄 Source {idx} - {meta['source']} (Page {meta['page']})"):
                    st.markdown(f'<div class="chunk-display">{chunk}</div>', unsafe_allow_html=True)

# الباقي زي ما هو (الـ info في النهاية)

else:
    st.info(f"📁 Add documents to folder: {os.path.abspath(DOCS_FOLDER)}")
    # ... باقي الـ info
