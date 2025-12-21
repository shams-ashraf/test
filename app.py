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
import base64

# OCR setup
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Settings
DOCUMENTS_FOLDER = "./documents"
MIN_WIDTH = 40
MIN_HEIGHT = 40
OUTPUT_FOLDER = "extracted_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

# Text cleaning
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
    raw_text = pytesseract.image_to_string(image, lang='eng+ara+deu')
    if not raw_text.strip():
        return ""
    structured_text = structure_text_into_paragraphs(raw_text)
    if '|' in structured_text or '\t' in structured_text or re.search(r'\d+\s+\w+\s+\d+', structured_text):
        structured_text = "📊 [Table content from image]\n\n" + structured_text
    return structured_text

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
    doc = fitz.open(file_path)
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            structured = structure_text_into_paragraphs(text)
            page_chunks = create_smart_chunks(structured)
            chunks.extend(page_chunks)
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            if image.width >= MIN_WIDTH and image.height >= MIN_HEIGHT:
                img_path = os.path.join(OUTPUT_FOLDER, f"page{page_num+1}_img{img_index+1}.png")
                image.save(img_path)
                img_text = extract_and_structure_text_from_image(image)
                if img_text.strip():
                    if '📊' in img_text:
                        tables_count += 1
                    chunks.append(img_text)
    doc.close()
    return chunks, tables_count

def extract_docx_text(file_path):
    chunks = []
    tables_count = 0
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
    if full_text:
        combined_text = "\n".join(full_text)
        structured = structure_text_into_paragraphs(combined_text)
        text_chunks = create_smart_chunks(structured)
        chunks.extend(text_chunks)
    for idx, table in enumerate(doc.tables, 1):
        table_text = extract_table_from_docx(table, idx)
        if table_text.strip():
            tables_count += 1
            chunks.append(table_text)
    return chunks, tables_count

def extract_txt_text(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    structured = structure_text_into_paragraphs(text)
    chunks = create_smart_chunks(structured)
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

def embed_chunks(chunks):
    client = chromadb.Client()
    collection_name = f"docs_{uuid.uuid4().hex[:8]}"
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="intfloat/multilingual-e5-large"
        )
    )
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        collection.add(
            documents=batch,
            ids=[f"chunk_{i+j}" for j in range(len(batch))],
            metadatas=[{"source": "Document"} for j in range(len(batch))]
        )
    return collection

# ============= Streamlit UI =============
def main():
    st.set_page_config(
        page_title="Document Processor",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Document Processor & Embedding Generator")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info("Processing documents from './Documents' folder")
        
        st.write(f"📁 Folder: `{DOCUMENTS_FOLDER}`")
        
        if st.button("🔄 Process All Documents", type="primary", use_container_width=True):
            st.session_state['process_folder'] = True
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("📊 Statistics")
        stats_placeholder = st.empty()
    
    with col1:
        st.subheader("📑 Processed Documents")
        
        # Auto-process on load or button click
        if 'process_folder' not in st.session_state:
            st.session_state['process_folder'] = True
        
        if st.session_state.get('process_folder', False):
            all_files = [os.path.join(DOCUMENTS_FOLDER, f) for f in os.listdir(DOCUMENTS_FOLDER) if f.split('.')[-1].lower() in ['pdf', 'docx', 'doc', 'txt']]
            
            if not all_files:
                st.warning("⚠️ No documents found in the Documents folder!")
                st.info("Please add PDF, DOCX, DOC, or TXT files to the './Documents' folder")
            else:
                all_chunks = []
                
                for file_path in all_files:
                    file_name = os.path.basename(file_path)
                    with st.expander(f"📄 {file_name}", expanded=True):
                        with st.spinner(f"Processing {file_name}..."):
                            chunks, tables_count = process_document(file_path)
                            all_chunks.extend(chunks)
                        
                        st.success(f"✅ Extracted {len(chunks)} chunks")
                        st.info(f"📊 Detected {tables_count} tables")
                        
                        # Display chunks
                        for idx, chunk in enumerate(chunks, 1):
                            with st.container():
                                st.markdown(f"**Chunk {idx}**")
                                
                                # Check if it's a table
                                if "📊" in chunk or "┌─" in chunk:
                                    st.code(chunk, language=None)
                                else:
                                    st.text_area(
                                        f"Content {idx}",
                                        chunk,
                                        height=200,
                                        key=f"{file_name}_{idx}",
                                        label_visibility="collapsed"
                                    )
                                st.markdown("---")
                
                # Update statistics
                with stats_placeholder.container():
                    st.metric("Total Files", len(all_files))
                    st.metric("Total Chunks", len(all_chunks))
                
                # Generate embeddings
                if all_chunks:
                    with st.spinner("Generating embeddings..."):
                        collection = embed_chunks(all_chunks)
                        st.success("✅ Embeddings generated successfully!")
                        st.balloons()
                
                st.session_state['process_folder'] = False
        
        else:
            st.info("👆 Click 'Process All Documents' button in the sidebar to start")

if __name__ == "__main__":
    main()
