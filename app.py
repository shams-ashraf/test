import os
import re
import uuid
import fitz  # PyMuPDF
import docx
from PIL import Image
import pytesseract
from io import BytesIO
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# ------------------ Settings ------------------
DOCUMENTS_FOLDER = "documents"
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ------------------ Session State Initialization ------------------
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {"messages": [], "created_at": ""}
    st.session_state.current_chat_id = chat_id
if "page_reload" not in st.session_state:
    st.session_state.page_reload = False

# ------------------ Text Preprocessing ------------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
    return text.strip()

def structure_text_into_paragraphs(text):
    if not text or not text.strip():
        return ""
    text = clean_text(text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    paragraphs, current_paragraph = [], []
    for i, line in enumerate(lines):
        words_in_line = line.split()
        if len(words_in_line) < 3 and not (line[0].isupper() or re.match(r'^[\d]+[\.\):]', line)):
            continue
        is_heading = (line.isupper() and len(words_in_line) <= 10) or (len(words_in_line) <= 6 and line[0].isupper() and line.endswith(':'))
        if is_heading:
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
            paragraphs.append(f"\n🔹 {line}\n")
            continue
        is_list_item = re.match(r'^[\d]+[\.\)]\s', line) or re.match(r'^[•\-\*]\s', line)
        if is_list_item:
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
            paragraphs.append(f"  {line}")
            continue
        current_paragraph.append(line)
        ends_with_punctuation = line.endswith(('.', '!', '?', '؟', '!', '。'))
        next_is_new_section = False
        if i < len(lines) - 1:
            next_line = lines[i+1]
            next_words = next_line.split()
            next_is_new_section = re.match(r'^[\d]+[\.\)]\s', next_line) or re.match(r'^[•\-\*]\s', next_line) or (len(next_words) <= 6 and next_line[0].isupper()) or next_line.isupper()
        if ends_with_punctuation or next_is_new_section or i == len(lines)-1:
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
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
    for i in range(0, len(words), chunk_size-overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if len(chunk.split()) >= 30:
            chunks.append(chunk)
    return chunks

# ------------------ Document Extraction ------------------
def extract_pdf_text(file_path):
    try:
        doc = fitz.open(file_path)
        full_text = []
        for page in doc:
            text = page.get_text()
            if len(text.strip()) < 50:
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = extract_and_structure_text_from_image(img)
                except:
                    text = ""
            text = clean_text(text)
            if text:
                full_text.append(text)
        doc.close()
        return create_smart_chunks(" ".join(full_text)), 0
    except:
        return [], 0

def extract_docx_text(file_path):
    doc = docx.Document(file_path)
    full_text = [clean_text(para.text) for para in doc.paragraphs if clean_text(para.text)]
    return create_smart_chunks(" ".join(full_text)), 0

def extract_txt_text(file_path):
    with open(file_path,'r',encoding='utf-8',errors='ignore') as f:
        text = f.read()
    text = clean_text(text)
    return create_smart_chunks(text), 0

def process_document(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == "pdf":
        return extract_pdf_text(file_path)
    elif ext in ["docx","doc"]:
        return extract_docx_text(file_path)
    elif ext == "txt":
        return extract_txt_text(file_path)
    return [],0

# ------------------ Embeddings ------------------
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
            metadatas=[{"source":"Document"} for j in range(len(batch))]
        )
    return collection

# ------------------ Retrieval ------------------
def retrieve_chunks(collection, query, top_k=5):
    if not collection or not query.strip():
        return []
    results = collection.query(query_texts=[query], n_results=top_k)
    chunks = results["documents"][0]
    unique_chunks = []
    seen = set()
    for chunk in chunks:
        normalized = chunk.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique_chunks.append(chunk)
    return unique_chunks

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Document Retrieval Chatbot", layout="wide")
st.sidebar.markdown("# 🗂️ Document Retrieval")

# New Chat
if st.sidebar.button("✚ New Chat"):
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {"messages": [], "created_at": ""}
    st.session_state.current_chat_id = chat_id
    st.session_state.page_reload = not st.session_state.page_reload

current_chat = st.session_state.chats[st.session_state.current_chat_id]

# ------------------ Load Documents with Progress ------------------
@st.cache_resource
def load_documents_and_embeddings_with_progress():
    all_files = [os.path.join(DOCUMENTS_FOLDER, f) for f in os.listdir(DOCUMENTS_FOLDER)]
    all_chunks = []
    if not all_files:
        st.warning("📂 No documents found in the folder.")
        return None
    progress_bar = st.progress(0)
    status_text = st.empty()
    for idx, file_path in enumerate(all_files,1):
        status_text.text(f"📄 Processing file {idx}/{len(all_files)}: {os.path.basename(file_path)}")
        chunks,_ = process_document(file_path)
        all_chunks.extend(chunks)
        progress_bar.progress(idx / len(all_files))
    if all_chunks:
        status_text.text("🧠 Generating embeddings for all chunks...")
        collection = embed_chunks(all_chunks)
        status_text.text("✅ All documents processed and embeddings generated!")
        progress_bar.empty()
        return collection
    status_text.text("⚠️ No chunks extracted from documents.")
    progress_bar.empty()
    return None

collection = load_documents_and_embeddings_with_progress()

# ------------------ Chat Interface ------------------
for msg in current_chat["messages"]:
    role = "You" if msg["role"]=="user" else "Assistant"
    st.markdown(f"**{role}:** {msg['content']}")

st.markdown("---")

user_input = st.text_input("Ask your question here...")
if user_input and collection:
    current_chat["messages"].append({"role":"user","content":user_input})
    with st.spinner("🔍 Retrieving relevant chunks..."):
        retrieved = retrieve_chunks(collection, user_input, top_k=5)
        answer_text = "\n\n".join(retrieved) if retrieved else "No relevant information found."
    current_chat["messages"].append({"role":"assistant","content":answer_text})
    # تحديث تلقائي بدون experimental_rerun
    st.session_state.page_reload = not st.session_state.page_reload
