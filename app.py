import os
import re
import io
import fitz  # PyMuPDF
import docx
from PIL import Image
import pytesseract
from io import BytesIO
import uuid
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# إعدادات الصفحة
st.set_page_config(
    page_title="معالج المستندات الذكي",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS مخصص لتحسين المظهر
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .chunk-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .table-container {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    .vector-info {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# OCR setup
try:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
except:
    pass

# Settings
DOCUMENTS_FOLDER = "./Documents"
MIN_WIDTH = 40
MIN_HEIGHT = 40
OUTPUT_FOLDER = "extracted_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

# Text cleaning functions
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
            paragraphs.append(f"  {line}")
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
    structured_text = ""
    for para in paragraphs:
        if para.startswith('\n🔹'):
            structured_text += para
        elif para.startswith('  '):
            structured_text += para + "\n"
        else:
            structured_text += para + "\n\n"
    return structured_text.strip() if structured_text else text

def extract_and_structure_text_from_image(image):
    try:
        raw_text = pytesseract.image_to_string(image, lang='eng+ara+deu')
        if not raw_text.strip():
            return ""
        structured_text = structure_text_into_paragraphs(raw_text)
        if '|' in structured_text or '\t' in structured_text or re.search(r'\d+\s+\w+\s+\d+', structured_text):
            structured_text = "📊 [Table content from image]\n\n" + structured_text
        return structured_text
    except:
        return ""

def create_smart_chunks(text, chunk_size=700, overlap=200):
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

def extract_table_from_docx(table, table_number=None):
    if len(table.rows) == 0:
        return ""
    headers = [clean_text(cell.text) for cell in table.rows[0].cells if cell.text.strip()]
    if not headers:
        return ""
    formatted_lines = []
    title = f"Table {table_number}" if table_number else "Table"
    formatted_lines.append(f"\n┌{'─'*58}┐\n│  📊 {title}{' '*(54-len(title))}│\n└{'─'*58}┘\n")
    formatted_lines.append("📋 Columns:")
    for idx, header in enumerate(headers, 1):
        formatted_lines.append(f"  {idx}. {header}")
    formatted_lines.append(f"\n{'─'*60}\n📊 Data:\n")
    row_count = 0
    for row in table.rows[1:]:
        cells = [clean_text(cell.text) for cell in row.cells]
        if not any(cells):
            continue
        row_count += 1
        formatted_lines.append(f"▸ Row {row_count}:")
        for header, value in zip(headers, cells):
            formatted_lines.append(f"  • {header}: {value if value else '[Empty]'}")
        formatted_lines.append("")
    formatted_lines.append(f"{'─'*60}\n📈 Summary: {row_count} rows, {len(headers)} columns\n{'─'*60}\n")
    return "\n".join(formatted_lines)

def extract_pdf_text(file_path):
    chunks = []
    tables_count = 0
    try:
        doc = fitz.open(file_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        structured = structure_text_into_paragraphs(full_text)
        chunks = create_smart_chunks(structured)
    except:
        pass
    return chunks, tables_count

def extract_docx_text(file_path):
    chunks = []
    tables_count = 0
    try:
        doc = docx.Document(file_path)
        full_text = ""
        for para in doc.paragraphs:
            full_text += para.text + "\n"
        tables_count = len(doc.tables)
        for idx, table in enumerate(doc.tables, 1):
            table_text = extract_table_from_docx(table, idx)
            if table_text:
                full_text += "\n" + table_text + "\n"
        structured = structure_text_into_paragraphs(full_text)
        chunks = create_smart_chunks(structured)
    except:
        pass
    return chunks, tables_count

def extract_txt_text(file_path):
    chunks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        structured = structure_text_into_paragraphs(text)
        chunks = create_smart_chunks(structured)
    except:
        pass
    return chunks, 0

def process_document(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_pdf_text(file_path)
    elif ext in ['docx', 'doc']:
        return extract_docx_text(file_path)
    elif ext == 'txt':
        return extract_txt_text(file_path)
    else:
        return [], 0

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("intfloat/multilingual-e5-large")

def embed_chunks(chunks, progress_bar=None):
    client = chromadb.Client()
    collection_name = f"docs_{uuid.uuid4().hex[:8]}"
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="intfloat/multilingual-e5-large"
        )
    )
    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        collection.add(
            documents=batch,
            ids=[f"chunk_{i+j}" for j in range(len(batch))],
            metadatas=[{"source": "Document", "chunk_index": i+j} for j in range(len(batch))]
        )
        if progress_bar:
            progress_bar.progress((i + batch_size) / len(chunks))
    
    return collection

# واجهة التطبيق
st.title("📚 معالج المستندات الذكي مع ChromaDB")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ الإعدادات")
    chunk_size = st.slider("حجم القطعة (Chunk Size)", 300, 1000, 700, 50)
    overlap = st.slider("التداخل (Overlap)", 50, 300, 200, 50)
    
    st.markdown("---")
    st.header("📤 رفع الملفات")
    uploaded_files = st.file_uploader(
        "اختر الملفات",
        type=['pdf', 'docx', 'doc', 'txt'],
        accept_multiple_files=True
    )
    
    process_button = st.button("🚀 معالجة الملفات", use_container_width=True)

# Main content
if process_button and uploaded_files:
    # حفظ الملفات المرفوعة
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DOCUMENTS_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ تم رفع {len(uploaded_files)} ملف بنجاح!")
    
    # معالجة المستندات
    st.header("📊 نتائج المعالجة")
    
    all_chunks = []
    total_tables = 0
    file_stats = []
    
    progress_text = st.empty()
    main_progress = st.progress(0)
    
    files = [os.path.join(DOCUMENTS_FOLDER, f) for f in os.listdir(DOCUMENTS_FOLDER)]
    
    for idx, file_path in enumerate(files):
        progress_text.text(f"⏳ معالجة الملف: {os.path.basename(file_path)}")
        main_progress.progress((idx + 1) / len(files))
        
        with st.expander(f"📄 {os.path.basename(file_path)}", expanded=True):
            chunks, tables_count = process_document(file_path)
            
            # إحصائيات الملف
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📝 عدد القطع", len(chunks))
            with col2:
                st.metric("📊 الجداول", tables_count)
            with col3:
                total_words = sum(len(chunk.split()) for chunk in chunks)
                st.metric("📖 الكلمات", f"{total_words:,}")
            with col4:
                file_size = os.path.getsize(file_path) / 1024
                st.metric("💾 الحجم", f"{file_size:.1f} KB")
            
            # عرض القطع
            st.markdown("### 📑 القطع المستخرجة")
            for chunk_idx, chunk in enumerate(chunks, 1):
                st.markdown(f"""
                <div class='chunk-container'>
                    <h4>📄 القطعة رقم {chunk_idx}</h4>
                    <p style='white-space: pre-wrap; font-family: monospace;'>{chunk}</p>
                    <small>📏 عدد الكلمات: {len(chunk.split())}</small>
                </div>
                """, unsafe_allow_html=True)
            
            all_chunks.extend(chunks)
            total_tables += tables_count
            file_stats.append({
                'اسم الملف': os.path.basename(file_path),
                'القطع': len(chunks),
                'الجداول': tables_count,
                'الكلمات': total_words
            })
    
    progress_text.text("✅ اكتملت معالجة جميع الملفات!")
    
    # ملخص عام
    st.markdown("---")
    st.header("📈 الملخص الإجمالي")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h2>{len(files)}</h2>
            <p>📁 إجمالي الملفات</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h2>{len(all_chunks)}</h2>
            <p>📝 إجمالي القطع</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h2>{total_tables}</h2>
            <p>📊 إجمالي الجداول</p>
        </div>
        """, unsafe_allow_html=True)
    
    # جدول الإحصائيات
    st.markdown("### 📋 جدول الإحصائيات التفصيلي")
    df_stats = pd.DataFrame(file_stats)
    st.dataframe(df_stats, use_container_width=True)
    
    # إنشاء Embeddings
    st.markdown("---")
    st.header("🧠 إنشاء Embeddings")
    
    with st.spinner("⏳ جاري إنشاء الـ Embeddings..."):
        embed_progress = st.progress(0)
        collection = embed_chunks(all_chunks, embed_progress)
        st.success("✅ تم إنشاء الـ Embeddings بنجاح!")
    
    # معلومات ChromaDB
    st.markdown("### 🗄️ معلومات قاعدة البيانات الشعاعية")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='vector-info'>
            <h4>📦 معلومات المجموعة</h4>
            <ul>
                <li><b>اسم المجموعة:</b> {collection.name}</li>
                <li><b>عدد الـ Vectors:</b> {collection.count()}</li>
                <li><b>نموذج Embedding:</b> multilingual-e5-large</li>
                <li><b>أبعاد Vector:</b> 1024</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='vector-info'>
            <h4>📊 إحصائيات Embedding</h4>
            <ul>
                <li><b>إجمالي Chunks:</b> {len(all_chunks)}</li>
                <li><b>حجم Chunk:</b> {chunk_size} كلمة</li>
                <li><b>التداخل:</b> {overlap} كلمة</li>
                <li><b>الوقت:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # عرض عينة من الـ Embeddings
    st.markdown("### 🔍 عينة من الـ Vectors")
    sample_results = collection.get(limit=3, include=['embeddings', 'documents', 'metadatas'])
    
    for i in range(min(3, len(sample_results['ids']))):
        with st.expander(f"Vector #{i+1} - {sample_results['ids'][i]}"):
            st.markdown(f"**النص:**")
            st.text(sample_results['documents'][i][:300] + "...")
            st.markdown(f"**Metadata:**")
            st.json(sample_results['metadatas'][i])
            st.markdown(f"**Vector (أول 10 قيم):**")
            vector_sample = sample_results['embeddings'][i][:10]
            st.code(f"[{', '.join([f'{v:.4f}' for v in vector_sample])}, ...]")
    
    # البحث التجريبي
    st.markdown("---")
    st.header("🔎 البحث التجريبي")
    query = st.text_input("أدخل استعلام البحث:")
    
    if query:
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        
        st.markdown("### 📊 نتائج البحث")
        for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
            similarity = 1 - distance
            st.markdown(f"""
            <div class='chunk-container'>
                <h4>🎯 النتيجة #{i+1} - التشابه: {similarity:.2%}</h4>
                <p>{doc[:400]}...</p>
            </div>
            """, unsafe_allow_html=True)

else:
    # صفحة البداية
    st.info("👈 ابدأ برفع الملفات من القائمة الجانبية")
    
    st.markdown("""
    ### 🌟 الميزات
    - 📄 دعم ملفات PDF, DOCX, TXT
    - 🔍 استخراج النصوص والجداول
    - 🧩 تقسيم ذكي للنصوص (Smart Chunking)
    - 🧠 إنشاء Embeddings باستخدام نموذج متعدد اللغات
    - 🗄️ تخزين في ChromaDB
    - 🔎 بحث دلالي (Semantic Search)
    - 📊 إحصائيات مفصلة
    
    ### 📖 كيفية الاستخدام
    1. ارفع ملف أو أكثر من القائمة الجانبية
    2. اضغط على "معالجة الملفات"
    3. شاهد النتائج والإحصائيات
    4. جرب البحث الدلالي!
    """)
