import streamlit as st
import re
import fitz
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

# Configuration - التعديل الرئيسي هنا
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
PDF_PASSWORD = "mbe2025"  # باسورد الملف المحمي تلقائي
DOCS_FOLDER = "/mount/src/test/documents"  # المسار الصحيح اللي الملفات فيه
CACHE_FOLDER = os.getenv("CACHE_FOLDER", "./cache")

os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found in environment variables!")

# Helper Functions
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
    text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
    return text.strip()

def structure_text_into_paragraphs(text):
    if not text or not text.strip():
        return ""
   
    text = clean_text(text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
   
    if not lines:
        return ""
   
    paragraphs = []
    current_paragraph = []
   
    for i, line in enumerate(lines):
        words_in_line = line.split()
       
        if len(words_in_line) < 3 and not (line[0].isupper() or re.match(r'^[\d]+[\.\):]', line)):
            continue
       
        is_heading = (
            (line.isupper() and len(words_in_line) <= 10) or
            (len(words_in_line) <= 6 and line[0].isupper() and line.endswith(':'))
        )
       
        if is_heading:
            if current_paragraph:
                paragraph_text = ' '.join(current_paragraph)
                paragraph_text = re.sub(r'\s+', ' ', paragraph_text)
                paragraph_text = re.sub(r'\s+([.,!?;:])', r'\1', paragraph_text)
                paragraphs.append(paragraph_text.strip())
                current_paragraph = []
            paragraphs.append(f"\n🔹 {line}\n")
            continue
       
        is_list_item = re.match(r'^[\d]+[\.\)]\s', line) or re.match(r'^[•\-\*]\s', line)
       
        if is_list_item:
            if current_paragraph:
                paragraph_text = ' '.join(current_paragraph)
                paragraph_text = re.sub(r'\s+', ' ', paragraph_text)
                paragraph_text = re.sub(r'\s+([.,!?;:])', r'\1', paragraph_text)
                paragraphs.append(paragraph_text.strip())
                current_paragraph = []
            paragraphs.append(f" {line}")
            continue
       
        current_paragraph.append(line)
       
        ends_with_punctuation = line.endswith(('.', '!', '?', '؟', '!', '。'))
        next_is_new_section = False
       
        if i < len(lines) - 1:
            next_line = lines[i + 1]
            next_words = next_line.split()
            next_is_new_section = (
                re.match(r'^[\d]+[\.\)]\s', next_line) or
                re.match(r'^[•\-\*]\s', next_line) or
                (len(next_words) <= 6 and next_line[0].isupper()) or
                next_line.isupper()
            )
       
        is_last_line = (i == len(lines) - 1)
       
        if (ends_with_punctuation or next_is_new_section or is_last_line):
            if current_paragraph:
                paragraph_text = ' '.join(current_paragraph)
                paragraph_text = re.sub(r'\s+', ' ', paragraph_text)
                paragraph_text = re.sub(r'\s+([.,!?;:])', r'\1', paragraph_text)
                paragraph_text = re.sub(r'([.,!?;:])\s*([.,!?;:])', r'\1', paragraph_text)
                paragraphs.append(paragraph_text.strip())
                current_paragraph = []
   
    if paragraphs:
        structured_text = ""
        for para in paragraphs:
            if para.startswith('\n🔹'):
                structured_text += para
            elif para.startswith(' '):
                structured_text += para + "\n"
            else:
                structured_text += para + "\n\n"
        return structured_text.strip()
   
    return text

def create_smart_chunks(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
   
    if len(words) <= chunk_size:
        return [text] if text.strip() else []
   
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        if len(chunk.split()) >= 30:
            chunks.append(chunk)
   
    return chunks

# تحسين تنسيق الجداول لـ Markdown (أهم حاجة للدقة)
def format_table_as_structured_text(extracted_table, table_number=None):
    if not extracted_table or len(extracted_table) == 0:
        return ""
   
    headers = [str(cell).strip() if cell else f"Column_{i+1}" for i, cell in enumerate(extracted_table[0])]
    headers = [clean_text(h) for h in headers]
   
    text = f"\n📊 Table {table_number or ''}\n\n"
    if headers:
        text += "| " + " | ".join(headers) + " |\n"
        text += "| " + " --- |" * len(headers) + " |\n"
    
    row_count = 0
    for row in extracted_table[1:]:
        cells = [str(cell).strip() if cell else "" for cell in row]
        cells = [clean_text(c) for c in cells]
        if any(cells):
            text += "| " + " | ".join(cells) + " |\n"
            row_count += 1
    
    text += f"\n**Summary**: {row_count} data rows, {len(headers)} columns.\n"
    return text

def extract_pdf_detailed(filepath):
    try:
        doc = fitz.open(filepath)
        if doc.is_encrypted:
            if not doc.authenticate(PDF_PASSWORD):
                doc.close()
                return None, "❌ Invalid PDF password"
    except Exception as e:
        return None, f"❌ Error opening PDF: {str(e)}"
   
    file_info = {
        'chunks': [],
        'total_pages': len(doc),
        'total_tables': 0,
        'pages_with_tables': [],
    }
   
    for page_num in range(len(doc)):
        page = doc[page_num]
        all_elements = []
       
        text_blocks = page.get_text("dict")["blocks"]
        for block in text_blocks:
            if block.get('type') == 0:
                y_pos = block.get('bbox', [0, 0, 0, 0])[1]
                text_content = ""
                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        text_content += span.get('text', '') + ' '
                if text_content.strip():
                    structured_content = structure_text_into_paragraphs(text_content)
                    all_elements.append({
                        'type': 'text',
                        'y_position': y_pos,
                        'content': structured_content
                    })
       
        tables = page.find_tables()
        if tables and len(tables.tables) > 0:
            file_info['pages_with_tables'].append(page_num + 1)
           
            for table_num, table in enumerate(tables.tables, 1):
                file_info['total_tables'] += 1
                extracted_table = table.extract()
                if extracted_table:
                    table_text = format_table_as_structured_text(extracted_table, file_info['total_tables'])
                    all_elements.append({
                        'type': 'table',
                        'y_position': table.bbox[1] if table.bbox else 0,
                        'content': table_text
                    })
       
        all_elements.sort(key=lambda x: x['y_position'])
       
        page_text = f"\n{'═' * 60}\n📄 Page {page_num + 1}\n{'═' * 60}\n\n"
        for element in all_elements:
            page_text += element['content'] + "\n\n"
       
        page_chunks = create_smart_chunks(page_text, chunk_size=1500, overlap=250)
        file_info['chunks'].extend(page_chunks)
   
    doc.close()
    return file_info, None

def extract_docx_detailed(filepath):
    doc = docx.Document(filepath)
    file_info = {
        'chunks': [],
        'total_pages': 1,
        'total_tables': 0,
        'pages_with_tables': [],
    }
   
    all_text = []
    table_counter = 0
   
    for element in doc.element.body:
        if element.tag.endswith('p'):
            for para in doc.paragraphs:
                if para._element == element:
                    text = clean_text(para.text)
                    if text:
                        structured = structure_text_into_paragraphs(text)
                        if structured:
                            all_text.append(structured)
                    break
       
        elif element.tag.endswith('tbl'):
            for table in doc.tables:
                if table._element == element:
                    file_info['total_tables'] += 1
                    table_counter += 1
                    table_text = format_table_as_structured_text(
                        [[cell.text for cell in row.cells] for row in table.rows],
                        table_counter
                    )
                    if table_text:
                        all_text.append(table_text)
                    break
   
    complete_text = "\n\n".join(all_text)
    file_info['chunks'] = create_smart_chunks(complete_text, chunk_size=1500, overlap=250)
   
    if file_info['total_tables'] > 0:
        file_info['pages_with_tables'] = [1]
   
    return file_info, None

def extract_txt_detailed(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    structured_text = structure_text_into_paragraphs(text)
    file_info = {
        'chunks': create_smart_chunks(structured_text, chunk_size=1500, overlap=250),
        'total_pages': 1,
        'total_tables': 0,
        'pages_with_tables': [],
    }
    return file_info, None

def get_files_from_folder():
    supported_extensions = ['*.pdf', '*.docx', '*.doc', '*.txt']
    files = []
    for ext in supported_extensions:
        files.extend(glob.glob(os.path.join(DOCS_FOLDER, ext)))
    return files

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large"
    )

def answer_question_with_groq(query, relevant_chunks):
    if not GROQ_API_KEY:
        return "❌ Please set GROQ_API_KEY in environment variables"
   
    context = "\n\n---\n\n".join(relevant_chunks[:8])  # زدت عدد الـ chunks لـ 8
   
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are an intelligent assistant specialized in analyzing biomedical engineering documents at Hochschule Anhalt.
CRITICAL RULES:
1. Answer ONLY from the provided context
2. Search thoroughly through ALL context before saying "no information"
3. Context may be in multiple languages (English, German) - check all
4. Use the SAME language as the question
5. Be CONCISE by default - keep answers short and direct
6. For table questions (e.g. how many modules): Count precisely and list them
7. If information not found: "No sufficient information in the available documents" """
            },
            {
                "role": "user",
                "content": f"""Context from documents:
{context}

Question: {query}

Answer precisely from the context only."""
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1500,
    }
   
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json=data,
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error connecting to Groq: {str(e)}"

# Main UI
st.markdown("""
<div class="main-card">
    <h1 style='text-align: center; margin: 0;'>📄 Smart Document Extractor</h1>
    <p style='text-align: center; margin-top: 10px;'>Automatic processing from documents folder</p>
</div>
""", unsafe_allow_html=True)

available_files = get_files_from_folder()
if not available_files:
    st.warning(f"⚠️ No documents found in folder: {DOCS_FOLDER}")
    st.info(f"📁 Please add PDF, DOCX, or TXT files to the folder above.")
else:
    st.success(f"✅ Found {len(available_files)} document(s) in folder")
    with st.expander("📂 Available Files", expanded=True):
        for file in available_files:
            st.write(f"• {os.path.basename(file)}")

if available_files and st.button("🚀 Process All Documents", type="primary", use_container_width=True):
    with st.spinner("Processing documents..."):
        files_data = {}
        all_chunks = []
        all_metadata = []
       
        client = chromadb.Client()
        collection_name = f"docs_{uuid.uuid4().hex[:8]}"
        collection = client.create_collection(
            name=collection_name,
            embedding_function=get_embedding_function()
        )
       
        progress_bar = st.progress(0)
        status_text = st.empty()
       
        for idx, filepath in enumerate(available_files):
            filename = os.path.basename(filepath)
            file_ext = filename.split('.')[-1].lower()
           
            status_text.text(f"Processing: {filename}...")
           
            file_hash = get_file_hash(filepath)
            cache_key = f"{file_hash}_{file_ext}"
            cached_data = load_cache(cache_key)
           
            if cached_data:
                st.info(f"📦 Using cached data for: {filename}")
                file_info = cached_data
                error = None
            else:
                if file_ext == 'pdf':
                    file_info, error = extract_pdf_detailed(filepath)
                elif file_ext in ['docx', 'doc']:
                    file_info, error = extract_docx_detailed(filepath)
                elif file_ext == 'txt':
                    file_info, error = extract_txt_detailed(filepath)
                else:
                    error = "Unsupported file type"
                    file_info = None
               
                if error:
                    st.error(f"❌ Error processing {filename}: {error}")
                    continue
               
                save_cache(cache_key, file_info)
                st.success(f"💾 Cached data for: {filename}")
           
            files_data[filename] = file_info
           
            for chunk in file_info['chunks']:
                all_chunks.append(chunk)
                all_metadata.append({"source": filename})
           
            progress_bar.progress((idx + 1) / len(available_files))
       
        status_text.text("Building search index...")
       
        if all_chunks:
            batch_size = 500
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i+batch_size]
                metadata_batch = all_metadata[i:i+batch_size]
                collection.add(
                    documents=batch,
                    ids=[f"chunk_{i+j}" for j in range(len(batch))],
                    metadatas=metadata_batch
                )
       
        st.session_state.files_data = files_data
        st.session_state.collection = collection
        st.session_state.processed = True
       
        status_text.empty()
        st.success("✅ Processing completed successfully!")
        st.balloons()

if st.session_state.processed:
    st.markdown("---")
   
    total_chunks = sum(len(info['chunks']) for info in st.session_state.files_data.values())
    total_tables = sum(info['total_tables'] for info in st.session_state.files_data.values())
   
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='stat-box'><h2 style='color: #667eea; margin: 0;'>{len(st.session_state.files_data)}</h2><p>Files</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='stat-box'><h2 style='color: #667eea; margin: 0;'>{total_chunks}</h2><p>Chunks</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='stat-box'><h2 style='color: #667eea; margin: 0;'>{total_tables}</h2><p>Tables</p></div>", unsafe_allow_html=True)
   
    st.markdown("---")
   
    st.subheader("📂 Select a file to view details")
    selected_file = st.selectbox("Processed files:", list(st.session_state.files_data.keys()))
   
    if selected_file:
        file_info = st.session_state.files_data[selected_file]
       
        st.markdown(f"""
<div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;'>
    <h3>📊 File Statistics: {selected_file}</h3>
    <p>📄 Pages: {file_info['total_pages']}</p>
    <p>📝 Chunks: {len(file_info['chunks'])}</p>
    <p>📊 Tables: {file_info['total_tables']}</p>
    {f"<p>📊 Pages with tables: {', '.join(map(str, file_info['pages_with_tables']))}</p>" if file_info['pages_with_tables'] else ""}
</div>
        """, unsafe_allow_html=True)
       
        st.markdown("---")
       
        st.subheader(f"📚 Extracted chunks from {selected_file}")
        chunks_per_page = 5
        total_pages = (len(file_info['chunks']) + chunks_per_page - 1) // chunks_per_page
        page = st.selectbox("Select page", range(1, total_pages + 1), key=f"page_{selected_file}")
       
        start_idx = (page - 1) * chunks_per_page
        end_idx = start_idx + chunks_per_page
       
        for idx, chunk in enumerate(file_info['chunks'][start_idx:end_idx], start_idx + 1):
            with st.expander(f"📄 Chunk #{idx} of {len(file_info['chunks'])}"):
                st.markdown(f'<div class="chunk-display">{chunk}</div>', unsafe_allow_html=True)
       
        st.markdown("---")
   
    st.subheader("🔍 Ask about documents")
    query = st.text_input("Enter your question here...")
   
    col_search1, col_search2 = st.columns([3, 1])
    with col_search1:
        search_only = st.checkbox("Search only (no AI answer)", value=False)
    with col_search2:
        num_results = st.selectbox("Number of results", [5, 10, 15, 20], index=1)  # افتراضي 10
   
    if query:
        with st.spinner("Searching..."):
            results = st.session_state.collection.query(
                query_texts=[query],
                n_results=num_results
            )
           
            if not search_only and GROQ_API_KEY:
                st.markdown("### 🤖 AI Answer:")
                with st.spinner("Generating answer..."):
                    answer = answer_question_with_groq(query, results["documents"][0])
                    st.markdown(f"""
<div class="answer-box">
    <h4 style='margin-top: 0;'>💡 Answer:</h4>
    {answer}
</div>
                    """, unsafe_allow_html=True)
           
            st.markdown("---")
            st.markdown("### 📄 Answer Sources:")
           
            for idx, (chunk, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
                with st.expander(f"📄 Source {idx} - from file: {metadata['source']}"):
                    st.markdown(f'<div class="chunk-display">{chunk}</div>', unsafe_allow_html=True)
else:
    st.info(f"📁 Add documents to folder: {DOCS_FOLDER}")
    st.markdown("""
    **Supported formats:**
    - 📄 PDF (password "mbe2025" supported)
    - 📝 DOCX/DOC
    - 📃 TXT
    """)
