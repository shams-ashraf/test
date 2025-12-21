import streamlit as st
import os
import re
import fitz
import io
import docx
from PIL import Image
import pytesseract
import uuid
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# Configuration
st.set_page_config(
    page_title="مستخرج المستندات الذكي",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chunk-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-right: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .stats-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.all_chunks = []
    st.session_state.collection = None
    st.session_state.total_tables = 0

# Helper Functions (simplified versions from your code)
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def structure_text_into_paragraphs(text):
    if not text or not text.strip():
        return ""
    text = clean_text(text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n\n'.join(lines)

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

def extract_pdf_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    all_text = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            all_text.append(text)
    doc.close()
    complete_text = "\n\n".join(all_text)
    return create_smart_chunks(complete_text, chunk_size=1500, overlap=250)

def extract_docx_text(file):
    doc = docx.Document(file)
    all_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            all_text.append(para.text)
    complete_text = "\n\n".join(all_text)
    return create_smart_chunks(complete_text, chunk_size=1500, overlap=250)

def extract_txt_text(file):
    text = file.read().decode('utf-8', errors='ignore')
    return create_smart_chunks(text, chunk_size=1500, overlap=250)

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large"
    )

# Main UI
st.markdown("""
<div class="main-header">
    <h1>📄 مستخرج المستندات الذكي</h1>
    <p>ارفع ملفاتك واحصل على نتائج مهيكلة بشكل احترافي</p>
</div>
""", unsafe_allow_html=True)

# File Upload
uploaded_files = st.file_uploader(
    "ارفع مستنداتك (PDF, DOCX, TXT)",
    type=['pdf', 'docx', 'doc', 'txt'],
    accept_multiple_files=True
)

if uploaded_files and st.button("🚀 ابدأ المعالجة", type="primary", use_container_width=True):
    with st.spinner("جاري معالجة المستندات..."):
        all_chunks = []
        all_metadata = []
        
        # Create vector DB
        client = chromadb.Client()
        collection_name = f"docs_{uuid.uuid4().hex[:8]}"
        collection = client.create_collection(
            name=collection_name,
            embedding_function=get_embedding_function()
        )
        
        progress_bar = st.progress(0)
        
        for idx, file in enumerate(uploaded_files):
            file_ext = file.name.split('.')[-1].lower()
            
            # Extract text based on file type
            if file_ext == 'pdf':
                chunks = extract_pdf_text(file)
            elif file_ext in ['docx', 'doc']:
                chunks = extract_docx_text(file)
            elif file_ext == 'txt':
                chunks = extract_txt_text(file)
            else:
                continue
            
            # Add to collection
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({"source": file.name})
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        # Batch insert into Chroma
        batch_size = 500
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            metadata_batch = all_metadata[i:i+batch_size]
            collection.add(
                documents=batch,
                ids=[f"chunk_{i+j}" for j in range(len(batch))],
                metadatas=metadata_batch
            )
        
        st.session_state.all_chunks = all_chunks
        st.session_state.collection = collection
        st.session_state.processed = True
        st.success("✅ تمت المعالجة بنجاح!")

# Display Results
if st.session_state.processed:
    st.markdown("---")
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stats-box">
            <h2>{len(st.session_state.all_chunks)}</h2>
            <p>إجمالي القطع المستخرجة</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_words = sum(len(chunk.split()) for chunk in st.session_state.all_chunks)
        st.markdown(f"""
        <div class="stats-box">
            <h2>{total_words:,}</h2>
            <p>إجمالي الكلمات</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_chunk_size = total_words // len(st.session_state.all_chunks) if st.session_state.all_chunks else 0
        st.markdown(f"""
        <div class="stats-box">
            <h2>{avg_chunk_size}</h2>
            <p>متوسط حجم القطعة</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Search Functionality
    st.subheader("🔍 البحث في المستندات")
    query = st.text_input("ابحث عن محتوى معين...")
    
    if query:
        results = st.session_state.collection.query(
            query_texts=[query],
            n_results=5
        )
        
        st.markdown("### نتائج البحث:")
        for idx, chunk in enumerate(results["documents"][0], 1):
            with st.expander(f"📄 نتيجة {idx}"):
                st.markdown(f'<div class="chunk-card">{chunk}</div>', unsafe_allow_html=True)
    
    # Display All Chunks
    st.markdown("---")
    st.subheader("📚 جميع القطع المستخرجة")
    
    # Pagination
    chunks_per_page = 10
    total_pages = (len(st.session_state.all_chunks) + chunks_per_page - 1) // chunks_per_page
    
    page = st.selectbox("اختر الصفحة", range(1, total_pages + 1))
    
    start_idx = (page - 1) * chunks_per_page
    end_idx = start_idx + chunks_per_page
    
    for idx, chunk in enumerate(st.session_state.all_chunks[start_idx:end_idx], start_idx + 1):
        with st.expander(f"📄 القطعة رقم {idx}"):
            st.markdown(f'<div class="chunk-card">{chunk}</div>', unsafe_allow_html=True)

else:
    st.info("👆 ارفع ملفاتك للبدء")
