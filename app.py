import os
import re
import io
import glob
import uuid
import time
from datetime import datetime

import fitz
import docx
from PIL import Image
import pytesseract
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# ------------------ SETTINGS ------------------
DOCUMENTS_FOLDER = "./documents"
OUTPUT_FOLDER = "./extracted_images"
PDF_PASSWORD = os.getenv("PDF_PASSWORD", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
MIN_WIDTH = 40
MIN_HEIGHT = 40

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------------ UTILITIES ------------------
def clean_text(text: str) -> str:
    text = re.sub(r'[^\w\s\u0600-\u06FF.,!?;:()\-\'\"]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def structure_text_into_paragraphs(text: str) -> str:
    text = clean_text(text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    paragraphs, current = [], []

    for i, line in enumerate(lines):
        words = line.split()
        if len(words) < 3 and not (line[0].isupper() or re.match(r'^[\d]+[\.\):]', line)):
            continue

        is_heading = (line.isupper() and len(words) <= 10) or (len(words) <= 6 and line[0].isupper() and line.endswith(':'))
        if is_heading:
            if current:
                paragraphs.append(' '.join(current))
                current = []
            paragraphs.append(f"\n🔹 {line}\n")
            continue

        is_list = re.match(r'^[\d]+[\.\)]\s', line) or re.match(r'^[•\-\*]\s', line)
        if is_list:
            if current:
                paragraphs.append(' '.join(current))
                current = []
            paragraphs.append(f"  {line}")
            continue

        current.append(line)
        ends_with_punc = line.endswith(('.', '!', '?', '؟'))
        next_new_section = False
        if i < len(lines)-1:
            next_line = lines[i+1]
            next_words = next_line.split()
            next_new_section = re.match(r'^[\d]+[\.\)]\s', next_line) or re.match(r'^[•\-\*]\s', next_line) or (len(next_words) <= 6 and next_line[0].isupper()) or next_line.isupper()
        is_last = i == len(lines)-1
        if ends_with_punc or next_new_section or is_last:
            if current:
                paragraphs.append(' '.join(current))
                current = []

    structured_text = ""
    for para in paragraphs:
        if para.startswith('\n🔹') or para.startswith('  '):
            structured_text += para + "\n"
        else:
            structured_text += para + "\n\n"
    return structured_text.strip()

def create_smart_chunks(text: str, chunk_size=700, overlap=200):
    words = text.split()
    chunks = []
    if len(words) <= chunk_size:
        return [text] if text.strip() else []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if len(chunk.split()) >= 30:
            chunks.append(chunk)
    return chunks

def extract_text_from_image(image: Image.Image) -> str:
    raw_text = pytesseract.image_to_string(image, lang='eng+ara+deu')
    if not raw_text.strip():
        return ""
    structured = structure_text_into_paragraphs(raw_text)
    return structured

def extract_table_from_docx(table, table_number=None) -> str:
    if len(table.rows) == 0:
        return ""
    headers = [clean_text(cell.text) for cell in table.rows[0].cells]
    if not headers:
        return ""
    formatted = []
    formatted.append(f"\n┌{'─'*58}┐")
    formatted.append(f"│  📊 Table {table_number if table_number else ''}{' '*(50-len(str(table_number or '')))}│")
    formatted.append(f"└{'─'*58}┘\n")
    formatted.append("📋 Columns:")
    for idx, h in enumerate(headers,1):
        formatted.append(f"  {idx}. {h}")
    formatted.append(f"\n{'─'*60}\n📊 Data:\n")
    row_count = 0
    for row in table.rows[1:]:
        cells = [clean_text(cell.text) for cell in row.cells]
        if not any(cells):
            continue
        row_count += 1
        formatted.append(f"▸ Row {row_count}:")
        for h, val in zip(headers, cells):
            formatted.append(f"  • {h}: {val if val else '[Empty]'}")
        formatted.append("")
    formatted.append(f"{'─'*60}")
    formatted.append(f"📈 Summary: {row_count} rows, {len(headers)} columns")
    formatted.append(f"{'─'*60}\n")
    return "\n".join(formatted)

# ------------------ DOCUMENT PROCESSING ------------------
def extract_pdf_text(file_path):
    doc = fitz.open(file_path)
    full_text = []
    for page_index, page in enumerate(doc):
        text_blocks = page.get_text("dict")["blocks"]
        images_info = []
        # Images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            if image.width >= MIN_WIDTH and image.height >= MIN_HEIGHT:
                image_name = f"page{page_index+1}_img{img_index+1}.png"
                image_path = os.path.join(OUTPUT_FOLDER, image_name)
                image.save(image_path)
                structured = extract_text_from_image(image)
                images_info.append({'y_position': 0, 'name': image_name, 'text': structured})

        # Text
        for block in text_blocks:
            if block.get('type') == 0:
                text_content = " ".join(span['text'] for line in block['lines'] for span in line['spans'])
                if text_content.strip():
                    structured_content = structure_text_into_paragraphs(text_content)
                    full_text.extend(create_smart_chunks(structured_content, chunk_size=1500, overlap=250))

    doc.close()
    return full_text, len(images_info)

def extract_docx_text(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for element in doc.element.body:
        if element.tag.endswith('p'):
            for para in doc.paragraphs:
                if para._element == element:
                    text = clean_text(para.text)
                    if text:
                        structured = structure_text_into_paragraphs(text)
                        full_text.extend(create_smart_chunks(structured, chunk_size=1500, overlap=250))
                    break
        elif element.tag.endswith('tbl'):
            for table in doc.tables:
                if table._element == element:
                    table_text = extract_table_from_docx(table)
                    if table_text:
                        full_text.extend(create_smart_chunks(table_text, chunk_size=1500, overlap=250))
                    break
    return full_text, len(doc.tables)

def extract_txt_text(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    structured = structure_text_into_paragraphs(text)
    return create_smart_chunks(structured, chunk_size=1500, overlap=250), 0

def process_document(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_pdf_text(file_path)
    elif ext in ['docx', 'doc']:
        return extract_docx_text(file_path)
    elif ext == 'txt':
        return extract_txt_text(file_path)
    return [], 0

def get_files_from_folder(folder_path):
    if not os.path.exists(folder_path):
        return []
    files = []
    for ext in ["*.pdf", "*.docx", "*.doc", "*.txt"]:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
        files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
    return files

# ------------------ EMBEDDING & CHROMA ------------------
@st.cache_resource(show_spinner=False)
def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-large")

@st.cache_resource(show_spinner=False)
def load_documents_and_embeddings():
    files = get_files_from_folder(DOCUMENTS_FOLDER)
    if not files:
        return None, [], 0
    client = chromadb.Client()
    collection_name = f"docs_{uuid.uuid4().hex[:8]}"
    collection = client.create_collection(name=collection_name, embedding_function=get_embedding_function())
    all_chunks, file_info = [], []

    for file_path in files:
        chunks, _ = process_document(file_path)
        if chunks:
            all_chunks.extend(chunks)
            file_info.append({"name": os.path.basename(file_path), "chunks": len(chunks), "path": file_path})

    if all_chunks:
        batch_size = 500
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            collection.add(
                documents=batch,
                ids=[f"chunk_{i+j}" for j in range(len(batch))],
                metadatas=[{"source": file_info[min(j, len(file_info)-1)]["name"]} for j in range(len(batch))]
            )
        return collection, file_info, len(all_chunks)
    return None, [], 0

def retrieve_relevant_chunks(collection, query, n_results=35):
    results = collection.query(query_texts=[query], n_results=n_results)
    chunks = results["documents"][0]
    unique_chunks = []
    seen = set()
    for c in chunks:
        c_norm = c.lower().strip()
        if c_norm not in seen:
            seen.add(c_norm)
            unique_chunks.append(c)
    return unique_chunks

def build_conversation_context(messages, max_history=3):
    context = []
    recent = messages[-(max_history*2):]
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        context.append(f"{role}: {msg['content']}")
    return "\n".join(context)

def ask_groq(question, context, conversation_history=None, retries=3):
    if not GROQ_API_KEY:
        return "Please add your Groq API Key"
    system_msg = "You are an intelligent assistant answering biomedical documents. Answer ONLY from context."
    conv_context = f"\n\nPrevious Conversation:\n{conversation_history}\n" if conversation_history else ""
    safe_context = context[:8000] if len(context) > 8000 else context
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Context:\n{safe_context}\n{conv_context}\nQuestion: {question}"}
    ]
    data = {"model": GROQ_MODEL, "messages": messages, "temperature":0.2, "max_tokens":800, "top_p":0.9}
    for attempt in range(retries):
        try:
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                 headers={"Authorization": f"Bearer {GROQ_API_KEY}","Content-Type":"application/json"},
                                 json=data, timeout=60)
            if resp.status_code == 429:
                time.sleep((2**attempt)+1)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except:
            time.sleep(2)
    return "Unable to get response"

# ------------------ STREAMLIT APP ------------------
def main():
    st.set_page_config(page_title="Biomedical Doc Chatbot", page_icon="🧬", layout="wide")
    if "chats" not in st.session_state:
        chat_id = str(uuid.uuid4())
        st.session_state.chats = {chat_id: {"messages": [], "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")}}
        st.session_state.current_chat_id = chat_id

    collection, loaded_files, total_chunks = load_documents_and_embeddings()

    with st.sidebar:
        st.markdown("# 🧬 BioMed Chat")
        if st.button("✚ New Chat", key="new_chat_btn", use_container_width=True):
            chat_id = str(uuid.uuid4())
            st.session_state.chats[chat_id] = {"messages": [], "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
            st.session_state.current_chat_id = chat_id
            st.experimental_rerun()

        st.markdown("### 💬 Chat History")
        if st.session_state.chats:
            for chat_id, chat_data in list(st.session_state.chats.items())[::-1]:
                preview = chat_data["messages"][0]["content"][:35]+"..." if chat_data["messages"] else "Empty chat"
                col1,col2 = st.columns([5,1])
                with col1:
                    if st.button(f"💬 {preview}", key=f"chat_{chat_id}", use_container_width=True):
                        st.session_state.current_chat_id = chat_id
                        st.experimental_rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{chat_id}"):
                        del st.session_state.chats[chat_id]
                        if st.session_state.current_chat_id == chat_id:
                            st.session_state.current_chat_id = None
                        st.experimental_rerun()

    st.markdown("# 🧬 Biomedical Document Chatbot")
    if not collection:
        st.error(f"No documents found in {DOCUMENTS_FOLDER}")
        st.info("Add PDF, DOCX, TXT files to the folder")
        return

    current_chat = st.session_state.chats[st.session_state.current_chat_id]
    for msg in current_chat["messages"]:
        role_icon = "🧑" if msg["role"]=="user" else "🤖"
        role_name = "You" if msg["role"]=="user" else "Assistant"
        st.markdown(f"<div class='chat-message'><b>{role_icon} {role_name}:</b> {msg['content']}</div>", unsafe_allow_html=True)

    st.markdown("---")
    user_input = st.chat_input("Ask your question here...")
    if user_input:
        current_chat["messages"].append({"role":"user","content":user_input})
        with st.spinner("Analyzing documents..."):
            relevant_chunks = retrieve_relevant_chunks(collection, user_input, n_results=35)
            selected_chunks = relevant_chunks[:20]
            context = "\n\n".join(selected_chunks)
            conversation_history = build_conversation_context(current_chat["messages"][:-1])
            answer = ask_groq(user_input, context, conversation_history)
            current_chat["messages"].append({"role":"assistant","content":answer})
        st.experimental_rerun()

if __name__ == "__main__":
    main()
