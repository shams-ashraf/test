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
        direction: rtl;
    }
    .stats-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .file-stats {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-right: 3px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.files_data = {}
    st.session_state.collection = None

MIN_WIDTH = 40
MIN_HEIGHT = 40

# Helper Functions
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def structure_text_into_paragraphs(text):
    if not text or not text.strip():
        return ""
    text = clean_text(text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    paragraphs = []
    current_paragraph = []
    
    for i, line in enumerate(lines):
        words_in_line = line.split()
        if len(words_in_line) < 3:
            continue
            
        current_paragraph.append(line)
        ends_with_punctuation = line.endswith(('.', '!', '?', '؟', '!', '。'))
        is_last_line = (i == len(lines) - 1)
        
        if ends_with_punctuation or is_last_line:
            if current_paragraph:
                paragraph_text = ' '.join(current_paragraph)
                paragraph_text = re.sub(r'\s+', ' ', paragraph_text)
                paragraphs.append(paragraph_text.strip())
                current_paragraph = []
    
    return '\n\n'.join(paragraphs) if paragraphs else text

def create_smart_chunks(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    if len(words) <= chunk_size:
        return [text] if text.strip() else []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        if len(chunk.split()) >= 20:
            chunks.append(chunk)
    return chunks

def format_table_as_structured_text(extracted_table, table_number=None):
    if not extracted_table or len(extracted_table) == 0:
        return ""
    headers = [str(cell).strip() if cell else "" for cell in extracted_table[0]]
    headers = [clean_text(h) if h else f"عمود_{i+1}" for i, h in enumerate(headers)]
    
    formatted_lines = []
    if table_number:
        formatted_lines.append(f"\n📊 جدول رقم {table_number}\n{'─' * 50}")
    else:
        formatted_lines.append(f"\n📊 جدول\n{'─' * 50}")
    
    formatted_lines.append("\n📋 الأعمدة: " + " | ".join(headers))
    formatted_lines.append("\n" + "─" * 50)
    
    row_count = 0
    for row_idx, row in enumerate(extracted_table[1:], 1):
        row_cells = [str(cell).strip() if cell else "" for cell in row]
        row_cells = [clean_text(cell) for cell in row_cells]
        if not any(row_cells):
            continue
        row_count += 1
        formatted_lines.append(f"\nصف {row_count}:")
        for header, value in zip(headers, row_cells):
            if value:
                formatted_lines.append(f"  • {header}: {value}")
    
    formatted_lines.append("\n" + "─" * 50 + "\n")
    return "\n".join(formatted_lines)

def extract_and_structure_text_from_image(image):
    raw_text = pytesseract.image_to_string(image, lang='eng+ara')
    if not raw_text.strip():
        return ""
    structured_text = structure_text_into_paragraphs(raw_text)
    return structured_text

def extract_pdf_detailed(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    
    file_info = {
        'chunks': [],
        'total_pages': len(doc),
        'total_tables': 0,
        'total_images': 0,
        'pages_with_tables': [],
        'pages_with_images': []
    }
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Collect all elements with positions
        all_elements = []
        
        # Extract text blocks
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
        
        # Extract images with OCR
        images = page.get_images(full=True)
        if images:
            file_info['pages_with_images'].append(page_num + 1)
            file_info['total_images'] += len(images)
            
            for img_index, img in enumerate(images):
                xref = img[0]
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    img_rect = img_rects[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    width, height = image.size
                    
                    if width >= MIN_WIDTH and height >= MIN_HEIGHT:
                        structured_text = extract_and_structure_text_from_image(image)
                        if structured_text:
                            all_elements.append({
                                'type': 'image',
                                'y_position': img_rect.y0,
                                'content': f"\n╔{'═' * 58}╗\n║  📷 صورة {img_index + 1} (أبعاد: {width}x{height}){' ' * (34 - len(str(width)) - len(str(height)))}║\n╚{'═' * 58}╝\n\n{structured_text}"
                            })
        
        # Extract tables
        tables = page.find_tables()
        if tables and len(tables.tables) > 0:
            file_info['pages_with_tables'].append(page_num + 1)
            for table_num, table in enumerate(tables.tables, 1):
                file_info['total_tables'] += 1
                table_bbox = table.bbox
                y_position = table_bbox[1] if table_bbox else 0
                extracted_table = table.extract()
                if extracted_table:
                    table_text = format_table_as_structured_text(extracted_table, file_info['total_tables'])
                    all_elements.append({
                        'type': 'table',
                        'y_position': y_position,
                        'content': table_text
                    })
        
        # Sort all elements by Y position
        all_elements.sort(key=lambda x: x['y_position'])
        
        # Build page text
        page_text = f"\n{'═' * 60}\n📄 صفحة {page_num + 1}\n{'═' * 60}\n\n"
        for element in all_elements:
            page_text += element['content'] + "\n\n"
        
        # Create chunks for this page
        page_chunks = create_smart_chunks(page_text, chunk_size=500, overlap=100)
        file_info['chunks'].extend(page_chunks)
    
    doc.close()
    return file_info

def extract_docx_detailed(file):
    doc = docx.Document(file)
    
    file_info = {
        'chunks': [],
        'total_pages': 1,
        'total_tables': 0,
        'total_images': 0,
        'pages_with_tables': [],
        'pages_with_images': []
    }
    
    all_text = []
    
    # Extract paragraphs and tables
    for element in doc.element.body:
        if element.tag.endswith('p'):
            for para in doc.paragraphs:
                if para._element == element:
                    text = clean_text(para.text)
                    if text:
                        structured = structure_text_into_paragraphs(text)
                        all_text.append(structured)
                    break
        elif element.tag.endswith('tbl'):
            for table in doc.tables:
                if table._element == element:
                    file_info['total_tables'] += 1
                    table_text = format_table_as_structured_text(
                        [[cell.text for cell in row.cells] for row in table.rows],
                        file_info['total_tables']
                    )
                    if table_text:
                        all_text.append(table_text)
                    break
    
    complete_text = "\n\n".join(all_text)
    file_info['chunks'] = create_smart_chunks(complete_text, chunk_size=500, overlap=100)
    
    if file_info['total_tables'] > 0:
        file_info['pages_with_tables'] = [1]
    
    return file_info

def extract_txt_detailed(file):
    text = file.read().decode('utf-8', errors='ignore')
    structured_text = structure_text_into_paragraphs(text)
    
    file_info = {
        'chunks': create_smart_chunks(structured_text, chunk_size=500, overlap=100),
        'total_pages': 1,
        'total_tables': 0,
        'total_images': 0,
        'pages_with_tables': [],
        'pages_with_images': []
    }
    
    return file_info

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large"
    )

# Main UI
st.markdown("""
<div class="main-header">
    <h1>📄 مستخرج المستندات الذكي</h1>
    <p>ارفع ملفاتك واحصل على تحليل شامل ومفصل</p>
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
        files_data = {}
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
            
            # Extract based on file type
            if file_ext == 'pdf':
                file_info = extract_pdf_detailed(file)
            elif file_ext in ['docx', 'doc']:
                file_info = extract_docx_detailed(file)
            elif file_ext == 'txt':
                file_info = extract_txt_detailed(file)
            else:
                continue
            
            files_data[file.name] = file_info
            
            # Add to collection
            for chunk in file_info['chunks']:
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
        
        st.session_state.files_data = files_data
        st.session_state.collection = collection
        st.session_state.processed = True
        st.success("✅ تمت المعالجة بنجاح!")

# Display Results
if st.session_state.processed:
    st.markdown("---")
    
    # Overall Statistics
    total_chunks = sum(len(info['chunks']) for info in st.session_state.files_data.values())
    total_tables = sum(info['total_tables'] for info in st.session_state.files_data.values())
    total_images = sum(info['total_images'] for info in st.session_state.files_data.values())
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stats-box">
            <h2>{len(st.session_state.files_data)}</h2>
            <p>ملف</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-box">
            <h2>{total_chunks}</h2>
            <p>قطعة نصية</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stats-box">
            <h2>{total_tables}</h2>
            <p>جدول</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stats-box">
            <h2>{total_images}</h2>
            <p>صورة</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File Selection
    st.subheader("📂 اختر ملف لعرض تفاصيله")
    selected_file = st.selectbox("الملفات المعالجة:", list(st.session_state.files_data.keys()))
    
    if selected_file:
        file_info = st.session_state.files_data[selected_file]
        
        # File Statistics
        st.markdown(f"""
        <div class="file-stats">
            <h3>📊 إحصائيات الملف: {selected_file}</h3>
            <p><strong>📄 عدد الصفحات:</strong> {file_info['total_pages']}</p>
            <p><strong>📝 عدد القطع:</strong> {len(file_info['chunks'])}</p>
            <p><strong>📊 عدد الجداول:</strong> {file_info['total_tables']}</p>
            <p><strong>📷 عدد الصور:</strong> {file_info['total_images']}</p>
            {f"<p><strong>📊 الصفحات التي تحتوي على جداول:</strong> {', '.join(map(str, file_info['pages_with_tables']))}</p>" if file_info['pages_with_tables'] else ""}
            {f"<p><strong>📷 الصفحات التي تحتوي على صور:</strong> {', '.join(map(str, file_info['pages_with_images']))}</p>" if file_info['pages_with_images'] else ""}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display Chunks
        st.subheader(f"📚 القطع المستخرجة من {selected_file}")
        
        # Pagination
        chunks_per_page = 5
        total_pages = (len(file_info['chunks']) + chunks_per_page - 1) // chunks_per_page
        
        page = st.selectbox("اختر الصفحة", range(1, total_pages + 1), key=f"page_{selected_file}")
        
        start_idx = (page - 1) * chunks_per_page
        end_idx = start_idx + chunks_per_page
        
        for idx, chunk in enumerate(file_info['chunks'][start_idx:end_idx], start_idx + 1):
            with st.expander(f"📄 القطعة رقم {idx} من {len(file_info['chunks'])}"):
                st.markdown(f'<div class="chunk-card">{chunk}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Search Functionality
    st.subheader("🔍 البحث في جميع المستندات")
    query = st.text_input("ابحث عن محتوى معين...")
    
    if query:
        results = st.session_state.collection.query(
            query_texts=[query],
            n_results=10
        )
        
        st.markdown("### نتائج البحث:")
        for idx, (chunk, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
            with st.expander(f"📄 نتيجة {idx} - من ملف: {metadata['source']}"):
                st.markdown(f'<div class="chunk-card">{chunk}</div>', unsafe_allow_html=True)

else:
    st.info("👆 ارفع ملفاتك للبدء")
