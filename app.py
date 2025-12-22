import streamlit as st
import re
import fitz
import io
import docx
import pytesseract
import uuid
import glob
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import requests
import json
from datetime import datetime
from langdetect import detect
from io import BytesIO
from PIL import Image
import os
import time

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
    .image-display {
        border: 2px solid #667eea;
        border-radius: 8px;
        padding: 10px;
        margin: 15px 0;
        background: #f0f4ff;
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

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found in environment variables!")

# Helper Functions
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
    return text.strip()

def detect_table_structure_in_image(image):
    """Enhanced table detection in images using OCR data analysis"""
    try:
        # Get detailed OCR data with bounding boxes
        ocr_data = pytesseract.image_to_data(image, lang='eng+ara', output_type=pytesseract.Output.DICT)
        
        # Analyze vertical and horizontal alignment
        x_positions = {}
        y_positions = {}
        
        for i, text in enumerate(ocr_data['text']):
            if text.strip():
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                
                # Group by approximate x positions (columns)
                x_key = round(x / 20) * 20
                if x_key not in x_positions:
                    x_positions[x_key] = []
                x_positions[x_key].append(text)
                
                # Group by approximate y positions (rows)
                y_key = round(y / 15) * 15
                if y_key not in y_positions:
                    y_positions[y_key] = []
                y_positions[y_key].append(text)
        
        # Table detection criteria
        has_multiple_columns = len(x_positions) >= 2
        has_multiple_rows = len(y_positions) >= 2
        has_aligned_structure = has_multiple_columns and has_multiple_rows
        
        return has_aligned_structure, x_positions, y_positions
    except:
        return False, {}, {}

def extract_table_from_image(image):
    """Extract structured table data from image using DBSCAN clustering."""
    try:
        # OCR مع bounding boxes
        ocr_data = pytesseract.image_to_data(image, lang='eng+ara', output_type=pytesseract.Output.DICT)
        
        words = []
        for i, text in enumerate(ocr_data['text']):
            if text.strip():
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                words.append({
                    "text": text.strip(),
                    "x": x + w/2,  # مركز الكلمة
                    "y": y + h/2
                })

        if not words:
            return None

        # تحويل للإحداثيات numpy
        x_coords = np.array([w['x'] for w in words]).reshape(-1, 1)
        y_coords = np.array([w['y'] for w in words]).reshape(-1, 1)

        # تجميع الأعمدة (x) وصفوف (y) باستخدام DBSCAN
        x_labels = DBSCAN(eps=15, min_samples=1).fit_predict(x_coords)
        y_labels = DBSCAN(eps=12, min_samples=1).fit_predict(y_coords)

        # ترتيب الأعمدة والصفوف
        unique_x = sorted(np.unique(x_labels))
        unique_y = sorted(np.unique(y_labels))

        table_data = []
        for y_lab in unique_y:
            row = []
            for x_lab in unique_x:
                # الكلمات في هذا الصف والعمود
                cell_words = [w['text'] for w, xl, yl in zip(words, x_labels, y_labels) if xl == x_lab and yl == y_lab]
                row.append(" ".join(cell_words).strip())
            if any(row):
                table_data.append(row)

        # شرط للتأكد أنه جدول فعلي
        if len(table_data) > 1 and len(table_data[0]) > 1:
            return table_data
        return None

    except Exception as e:
        print("Error extracting table:", e)
        return None

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
    
    if paragraphs:
        structured_text = ""
        for para in paragraphs:
            if para.startswith('\n🔹'):
                structured_text += para
            elif para.startswith('  '):
                structured_text += para + "\n"
            else:
                structured_text += para + "\n\n"
        return structured_text.strip()
    
    return text

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

def format_table_as_structured_text(extracted_table, table_number=None):
    if not extracted_table or len(extracted_table) == 0:
        return ""
    
    headers = [str(cell).strip() if cell else "" for cell in extracted_table[0]]
    headers = [clean_text(h) if h else f"Column_{i+1}" for i, h in enumerate(headers)]
    
    if not headers:
        return ""
    
    formatted_lines = []
    
    if table_number:
        formatted_lines.append(f"\n┌{'─' * 58}┐")
        formatted_lines.append(f"│ 📊 Table #{table_number}{' ' * (54 - len(str(table_number)))}│")
        formatted_lines.append(f"└{'─' * 58}┘\n")
    else:
        formatted_lines.append(f"\n┌{'─' * 58}┐")
        formatted_lines.append(f"│ 📊 Table{' ' * 50}│")
        formatted_lines.append(f"└{'─' * 58}┘\n")
    
    formatted_lines.append("📋 Table Columns:")
    for idx, header in enumerate(headers, 1):
        formatted_lines.append(f"  {idx}. {header}")
    
    formatted_lines.append(f"\n{'─' * 60}\n")
    formatted_lines.append("📊 Table Data:\n")
    
    row_count = 0
    for row_idx, row in enumerate(extracted_table[1:], 1):
        row_cells = [str(cell).strip() if cell else "" for cell in row]
        row_cells = [clean_text(cell) for cell in row_cells]
        
        if not any(row_cells):
            continue
        
        row_count += 1
        formatted_lines.append(f"▸ Row #{row_count}:")
        for header, value in zip(headers, row_cells):
            if value:
                formatted_lines.append(f"  • {header}: {value}")
            else:
                formatted_lines.append(f"  • {header}: [Empty]")
        formatted_lines.append("")
    
    formatted_lines.append(f"{'─' * 60}")
    formatted_lines.append(f"📈 Summary: Table contains {row_count} rows and {len(headers)} columns")
    formatted_lines.append(f"{'─' * 60}\n")
    
    return "\n".join(formatted_lines)

def extract_and_structure_text_from_image(image, page_num, img_index):
    """Enhanced image text extraction with table detection"""
    width, height = image.size
    
    # First check if image contains a table structure
    table_data = extract_table_from_image(image)
    
    if table_data:
        # Image contains a table - format it as structured table
        table_text = format_table_as_structured_text(table_data)
        return f"\n╔{'═' * 58}╗\n║ 📊 Table extracted from image (Page {page_num}, Image {img_index}){' ' * 10}║\n║ Dimensions: {width}x{height}px{' ' * (45 - len(str(width)) - len(str(height)))}║\n╚{'═' * 58}╝\n\n{table_text}\n"
    
    # Not a table - extract as regular text
    raw_text = pytesseract.image_to_string(image, lang='eng+ara+deu')
    
    if not raw_text.strip():
        return ""
    
    structured_text = structure_text_into_paragraphs(raw_text)
    
    if structured_text:
        return f"\n╔{'═' * 58}╗\n║ 📷 Content extracted from image (Page {page_num}, Image {img_index}){' ' * 8}║\n║ Dimensions: {width}x{height}px{' ' * (45 - len(str(width)) - len(str(height)))}║\n╚{'═' * 58}╝\n\n{structured_text}\n"
    
    return ""

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
        
        # Extract images with enhanced OCR
        images = page.get_images(full=True)
        if images:
            file_info['pages_with_images'].append(page_num + 1)
            file_info['total_images'] += len(images)
            
            for img_index, img in enumerate(images, 1):
                xref = img[0]
                img_rects = page.get_image_rects(xref)
                
                if img_rects:
                    img_rect = img_rects[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    width, height = image.size
                    
                    if width >= MIN_WIDTH and height >= MIN_HEIGHT:
                        structured_text = extract_and_structure_text_from_image(
                            image, 
                            page_num + 1, 
                            img_index
                        )
                        
                        if structured_text:
                            all_elements.append({
                                'type': 'image',
                                'y_position': img_rect.y0,
                                'content': structured_text
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
                    table_text = format_table_as_structured_text(
                        extracted_table, 
                        file_info['total_tables']
                    )
                    all_elements.append({
                        'type': 'table',
                        'y_position': y_position,
                        'content': table_text
                    })
        
        # Sort all elements by Y position to maintain document order
        all_elements.sort(key=lambda x: x['y_position'])
        
        # Build page text
        page_text = f"\n{'═' * 60}\n📄 Page {page_num + 1}\n{'═' * 60}\n\n"
        for element in all_elements:
            page_text += element['content'] + "\n\n"
        
        # Create chunks for this page
        page_chunks = create_smart_chunks(page_text, chunk_size=1500, overlap=250)
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
    table_counter = 0
    
    # Extract paragraphs and tables in order
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
    
    return file_info

def extract_txt_detailed(file):
    text = file.read().decode('utf-8', errors='ignore')
    structured_text = structure_text_into_paragraphs(text)
    
    file_info = {
        'chunks': create_smart_chunks(structured_text, chunk_size=1500, overlap=250),
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

def answer_question_with_groq(query, relevant_chunks):
    if not GROQ_API_KEY:
        return "❌ Please set GROQ_API_KEY in environment variables"
    
    context = "\n\n---\n\n".join(relevant_chunks[:5])
    
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are a highly intelligent assistant. You will be given text extracted from documents, which may include tables, lists, headings, or unstructured data. Your task is:

                    1. Carefully read and understand the entire text. Analyze any tables, lists, or headings. Detect columns, rows, and relationships in tables if present.
                    2. For any question asked, think step by step:
                       a. Identify the relevant parts of the text.
                       b. Understand the structure and meaning before answering.
                       c. Check for keywords, headings, and notes that help clarify the answer.
                    3. Only answer based on the given text. Do not use external knowledge.
                    5. Give clear, structured answers. If the answer is a list, table, or explanation, format it cleanly.
                    6. Always double-check your reasoning before giving the final answer.
                    """
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1500,
        "top_p": 0.9
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
    <p style='text-align: center; margin-top: 10px;'>Upload your files and get comprehensive analysis</p>
</div>
""", unsafe_allow_html=True)

# File Upload
uploaded_files = st.file_uploader(
    "Upload your documents (PDF, DOCX, TXT)",
    type=['pdf', 'docx', 'doc', 'txt'],
    accept_multiple_files=True
)

if uploaded_files and st.button("🚀 Start Processing", type="primary", use_container_width=True):
    with st.spinner("Processing documents..."):
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
        st.success("✅ Processing completed successfully!")

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
<div class="stat-box">
    <h2 style='color: #667eea; margin: 0;'>{len(st.session_state.files_data)}</h2>
    <p style='margin: 5px 0 0 0;'>Files</p>
</div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
<div class="stat-box">
    <h2 style='color: #667eea; margin: 0;'>{total_chunks}</h2>
    <p style='margin: 5px 0 0 0;'>Chunks</p>
</div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
<div class="stat-box">
    <h2 style='color: #667eea; margin: 0;'>{total_tables}</h2>
    <p style='margin: 5px 0 0 0;'>Tables</p>
</div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
<div class="stat-box">
    <h2 style='color: #667eea; margin: 0;'>{total_images}</h2>
    <p style='margin: 5px 0 0 0;'>Images</p>
</div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File Selection
    st.subheader("📂 Select a file to view details")
    selected_file = st.selectbox("Processed files:", list(st.session_state.files_data.keys()))
    
    if selected_file:
        file_info = st.session_state.files_data[selected_file]
        
        # File Statistics
        st.markdown(f"""
<div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;'>
    <h3>📊 File Statistics: {selected_file}</h3>
    <p>📄 Pages: {file_info['total_pages']}</p>
    <p>📝 Chunks: {len(file_info['chunks'])}</p>
    <p>📊 Tables: {file_info['total_tables']}</p>
    <p>📷 Images: {file_info['total_images']}</p>
    {f"<p>📊 Pages with tables: {', '.join(map(str, file_info['pages_with_tables']))}</p>" if file_info['pages_with_tables'] else ""}
    {f"<p>📷 Pages with images: {', '.join(map(str, file_info['pages_with_images']))}</p>" if file_info['pages_with_images'] else ""}
</div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display Chunks
        st.subheader(f"📚 Extracted chunks from {selected_file}")
        
        # Pagination
        chunks_per_page = 5
        total_pages = (len(file_info['chunks']) + chunks_per_page - 1) // chunks_per_page
        page = st.selectbox("Select page", range(1, total_pages + 1), key=f"page_{selected_file}")
        
        start_idx = (page - 1) * chunks_per_page
        end_idx = start_idx + chunks_per_page
        
        for idx, chunk in enumerate(file_info['chunks'][start_idx:end_idx], start_idx + 1):
            with st.expander(f"📄 Chunk #{idx} of {len(file_info['chunks'])}"):
                st.markdown(f'<div class="chunk-display">{chunk}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Search Functionality with Groq
    st.subheader("🔍 Ask about documents")
    query = st.text_input("Enter your question here...")
    
    col_search1, col_search2 = st.columns([3, 1])
    with col_search1:
        search_only = st.checkbox("Search only (no AI answer)", value=False)
    with col_search2:
        num_results = st.selectbox("Number of results", [5, 10, 15, 20], index=0)
    
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
    st.info("👆 Upload your files to start")
