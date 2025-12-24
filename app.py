import streamlit as st
import re
import fitz
import docx
import uuid
import glob
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import requests
import os
import pickle
import hashlib
from styles import load_custom_css

st.set_page_config(
    page_title="BioMed Doc Chat",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_custom_css()

if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.files_data = {}
    st.session_state.collection = None
    st.session_state.processing_started = False

if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = {}
    st.session_state.current_chat_id = None
    st.session_state.next_chat_number = 1

if st.session_state.current_chat_id is None:
    chat_id = f"chat_{st.session_state.next_chat_number}"
    st.session_state.chat_sessions[chat_id] = {
        'messages': [],
        'name': f"Chat {st.session_state.next_chat_number}"
    }
    st.session_state.current_chat_id = chat_id
    st.session_state.next_chat_number += 1

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
PDF_PASSWORD = "mbe2025"
DOCS_FOLDER = "/mount/src/test/documents"
CACHE_FOLDER = os.getenv("CACHE_FOLDER", "./cache")

os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)

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
        pass

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

def create_smart_chunks(text, chunk_size=1000, overlap=200, page_num=None, source_file=None, is_table=False, table_num=None):
    words = text.split()
    chunks = []
    
    metadata = {
        'page': str(page_num) if page_num is not None else "N/A",
        'source': source_file if source_file else "Unknown",
        'is_table': str(is_table),
        'table_number': str(table_num) if table_num is not None else "N/A"
    }
   
    if len(words) <= chunk_size:
        if text.strip():
            return [{
                'content': text,
                'metadata': metadata
            }]
        return []
   
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        if len(chunk.split()) >= 30:
            chunks.append({
                'content': chunk,
                'metadata': metadata.copy()
            })
   
    return chunks

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
   
    filename = os.path.basename(filepath)
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
                        'content': structured_content,
                        'page': page_num + 1
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
                        'content': table_text,
                        'page': page_num + 1,
                        'table_num': file_info['total_tables']
                    })
       
        all_elements.sort(key=lambda x: x['y_position'])
       
        page_text = f"\n# Document: {filename} - Official MBE Regulations\n\n"
        page_text += f"\n{'═' * 60}\n📄 Page {page_num + 1}\n{'═' * 60}\n\n"
        for element in all_elements:
            page_text += element['content'] + "\n\n"
       
        page_chunks = create_smart_chunks(
            page_text, 
            chunk_size=1500, 
            overlap=250,
            page_num=page_num + 1,
            source_file=filename,
            is_table=False
        )
        
        for element in all_elements:
            if element['type'] == 'table':
                table_chunks = create_smart_chunks(
                    element['content'],
                    chunk_size=2000,
                    overlap=0,
                    page_num=element['page'],
                    source_file=filename,
                    is_table=True,
                    table_num=element.get('table_num')
                )
                file_info['chunks'].extend(table_chunks)
        
        file_info['chunks'].extend(page_chunks)
   
    doc.close()
    return file_info, None

def extract_docx_detailed(filepath):
    doc = docx.Document(filepath)
    filename = os.path.basename(filepath)
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
                        table_chunks = create_smart_chunks(
                            table_text,
                            chunk_size=2000,
                            overlap=0,
                            page_num=1,
                            source_file=filename,
                            is_table=True,
                            table_num=table_counter
                        )
                        file_info['chunks'].extend(table_chunks)
                    break
   
    complete_text = "\n\n".join(all_text)
    text_chunks = create_smart_chunks(
        complete_text, 
        chunk_size=1500, 
        overlap=250,
        page_num=1,
        source_file=filename
    )
    file_info['chunks'].extend(text_chunks)
   
    if file_info['total_tables'] > 0:
        file_info['pages_with_tables'] = [1]
   
    return file_info, None

def extract_txt_detailed(filepath):
    filename = os.path.basename(filepath)
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    structured_text = structure_text_into_paragraphs(text)
    chunks = create_smart_chunks(
        structured_text, 
        chunk_size=1500, 
        overlap=250,
        page_num=1,
        source_file=filename
    )
    file_info = {
        'chunks': chunks,
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

def answer_question_with_groq(query, relevant_chunks, chat_history=None):
    if not GROQ_API_KEY:
        return "❌ Please set GROQ_API_KEY in environment variables"
   
    context_parts = []
    for i, chunk_data in enumerate(relevant_chunks[:10], 1):
        content = chunk_data['content']
        meta = chunk_data['metadata']
        
        source = meta.get('source', 'Unknown')
        page = meta.get('page', 'N/A')
        is_table = meta.get('is_table', 'False')
        table_num = meta.get('table_number', 'N/A')
        
        citation = f"[Source {i}: {source}, Page {page}"
        if is_table == 'True' or is_table == True:
            citation += f", Table {table_num}"
        citation += "]"
        
        context_parts.append(f"{citation}\n{content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    conversation_summary = ""
    if chat_history and len(chat_history) > 0:
        recent = chat_history[-6:]
        conv_lines = []
        for msg in recent:
            role = "User" if msg['role'] == 'user' else "Assistant"
            content_preview = msg['content'][:300]
            conv_lines.append(f"{role}: {content_preview}")
        conversation_summary = "\n".join(conv_lines)
   
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are a precise MBE Document Assistant at Hochschule Anhalt specializing in Biomedical Engineering regulations.

CRITICAL RULES:
1. Answer ONLY from provided sources OR previous conversation if it's a follow-up question
2. ALWAYS cite sources: [Source X, Page Y] or [Source X, Page Y, Table Z]
3. For follow-up questions like "summarize", "tell me more", "explain that", or "what about that":
   - Check the conversation history FIRST
   - Summarize or expand on your PREVIOUS answer
   - Don't search for new information if the question refers to what you just said
4. If user says "summarize that" or "summarize it": Condense your LAST answer (from conversation history)
5. If no relevant info in sources OR history: "No sufficient information in the available documents"
6. Use the SAME language as the question:
   - English question → English answer
   - German question → German answer (Deutsch)
7. Be CONCISE - short, direct answers unless asked to elaborate
8. For counting questions: Count precisely and list all items with citations

LANGUAGE DETECTION:
- Detect question language automatically
- Respond in the exact same language
- Maintain professional academic tone

FOLLOW-UP DETECTION:
- "that", "it", "this", "summarize", "tell me more", "elaborate", "explain further" → Use conversation history
- New factual questions → Use sources

Remember: You're helping MBE students understand their program requirements clearly and accurately in their preferred language."""
            },
            {
                "role": "user",
                "content": f"""CONVERSATION HISTORY (use for follow-up questions):
{conversation_summary if conversation_summary else "No previous conversation"}

DOCUMENT SOURCES (use for new factual questions):
{context}

CURRENT QUESTION: {query}

Instructions: 
- If this is a follow-up (summarize/elaborate/that/it), answer from conversation history
- If this is a new question, answer from sources with citations
- Always respond in the SAME language as the question
- Always be precise and cite your sources

ANSWER:"""
            }
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
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

def process_documents_automatically():
    if st.session_state.processed or st.session_state.processing_started:
        return
    
    st.session_state.processing_started = True
    
    available_files = get_files_from_folder()
    if not available_files:
        st.session_state.processed = True
        return
    
    # Create progress placeholder
    progress_container = st.empty()
    
    files_data = {}
    all_chunks = []
    all_metadata = []
    
    client = chromadb.Client()
    collection_name = f"docs_{uuid.uuid4().hex[:8]}"
    collection = client.create_collection(
        name=collection_name,
        embedding_function=get_embedding_function()
    )
    
    total_files = len(available_files)
       
    for idx, filepath in enumerate(available_files, 1):
        filename = os.path.basename(filepath)
        file_ext = filename.split('.')[-1].lower()
        
        # Update progress
        percentage = int((idx / total_files) * 100)
        progress_container.markdown(f"""
        <div style='text-align: center; padding: 30px;'>
            <h3 style='color: #00d4ff;'>🔄 Loading Documents... {percentage}%</h3>
            <p style='color: #e8e8e8; font-size: 1.1rem;'>Processing: <strong>{filename}</strong></p>
            <div style='background: rgba(255,255,255,0.1); border-radius: 10px; height: 30px; margin: 20px auto; max-width: 500px; overflow: hidden;'>
                <div style='background: linear-gradient(90deg, #00d9ff 0%, #0099cc 100%); height: 100%; width: {percentage}%; transition: width 0.3s;'></div>
            </div>
            <p style='color: #a0a0a0;'>Document {idx} of {total_files}</p>
        </div>
        """, unsafe_allow_html=True)
        
        file_hash = get_file_hash(filepath)
        cache_key = f"{file_hash}_{file_ext}"
        cached_data = load_cache(cache_key)
        
        if cached_data:
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
                file_info, error = None, f"❌ Unsupported file type: {file_ext}"
            
            if file_info and not error:
                save_cache(cache_key, file_info)
        
        if error:
            continue
        
        if file_info and len(file_info['chunks']) > 0:
            files_data[filename] = file_info
            
            for chunk in file_info['chunks']:
                all_chunks.append(chunk['content'])
                all_metadata.append(chunk['metadata'])
    
    # Clear progress and finalize
    progress_container.empty()
    
    if all_chunks:
        chunk_ids = [f"chunk_{uuid.uuid4().hex}" for _ in all_chunks]
        collection.add(
            documents=all_chunks,
            metadatas=all_metadata,
            ids=chunk_ids
        )
        
        st.session_state.files_data = files_data
        st.session_state.collection = collection
        st.session_state.processed = True
    else:
        st.session_state.processed = True

st.markdown("""
<div style='text-align: center; margin-bottom: 30px;'>
    <h1 style='color: #00d4ff; font-size: 3em; margin: 0;'>
        🧬 Biomedical Document Chatbot
    </h1>
</div>
""", unsafe_allow_html=True)

process_documents_automatically()

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <div style='font-size: 4em;'>🧬</div>
        <h2 style='color: #00d4ff; margin: 10px 0;'>BioMed<br>Doc Chat</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
        chat_id = f"chat_{st.session_state.next_chat_number}"
        st.session_state.chat_sessions[chat_id] = {
            'messages': [],
            'name': f"Chat {st.session_state.next_chat_number}"
        }
        st.session_state.current_chat_id = chat_id
        st.session_state.next_chat_number += 1
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 💬 Chat History")
    
    for chat_id in list(st.session_state.chat_sessions.keys()):
        chat_name = st.session_state.chat_sessions[chat_id]['name']
        msg_count = len(st.session_state.chat_sessions[chat_id]['messages'])
        
        if msg_count > 0:
            first_msg = st.session_state.chat_sessions[chat_id]['messages'][0]['content']
            preview = first_msg[:50] + "..." if len(first_msg) > 50 else first_msg
        else:
            preview = "New conversation"
        
        if st.button(
            f"💬 {preview}",
            key=f"chat_{chat_id}",
            use_container_width=True,
            disabled=(chat_id == st.session_state.current_chat_id)
        ):
            st.session_state.current_chat_id = chat_id
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 📚 Document Information")
    
    if st.session_state.processed and st.session_state.files_data:
        with st.expander("📄 About This Chatbot", expanded=False):
            st.markdown(f"""
            **Documents Loaded:** {len(st.session_state.files_data)}
            
            **Capabilities:**
            - 🔍 Search through MBE documents
            - 💬 Ask questions in English or German
            - 📊 Extract information from tables
            - 📝 Get cited, accurate answers
            """)
            
            for filename, info in st.session_state.files_data.items():
                st.markdown(f"**{filename}**")
                st.text(f"📄 Pages: {info['total_pages']} | 📊 Tables: {info['total_tables']} | 📦 Chunks: {len(info['chunks'])}")
    else:
        with st.expander("📄 About This Chatbot", expanded=False):
            st.info("No documents loaded yet")

# Get current chat messages
current_messages = st.session_state.chat_sessions[st.session_state.current_chat_id]['messages']

if st.session_state.processed:
    for message in current_messages:
        role_class = "user-message" if message["role"] == "user" else "assistant-message"
        icon = "👤" if message["role"] == "user" else "🤖"
        header = "YOU" if message["role"] == "user" else "ASSISTANT"
        
        st.markdown(f"""
        <div class='chat-message {role_class}'>
            <div class='message-header'>{icon} {header}</div>
            <div>{message['content']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    if query := st.chat_input("Ask your question here..."):
        current_messages.append({"role": "user", "content": query})
        
        st.markdown(f"""
        <div class='chat-message user-message'>
            <div class='message-header'>👤 YOU</div>
            <div>{query}</div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🔍 Searching documents..."):
            try:
                results = st.session_state.collection.query(
                    query_texts=[query],
                    n_results=10
                )
                
                relevant_chunks = []
                if results['documents'] and len(results['documents'][0]) > 0:
                    for i in range(len(results['documents'][0])):
                        relevant_chunks.append({
                            'content': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i]
                        })
                
                answer = answer_question_with_groq(
                    query, 
                    relevant_chunks, 
                    chat_history=current_messages[:-1]
                )
                
                current_messages.append({"role": "assistant", "content": answer})
                
                st.markdown(f"""
                <div class='chat-message assistant-message'>
                    <div class='message-header'>🤖 ASSISTANT</div>
                    <div>{answer}</div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                error_msg = f"❌ Error processing query: {str(e)}"
                st.error(error_msg)
                current_messages.append({"role": "assistant", "content": error_msg})
        
        st.rerun()
else:
    # This will only show on very first load before processing starts
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h3 style='color: #00d4ff;'>🔄 Initializing system...</h3>
        <p style='color: #e8e8e8;'>Please wait a moment</p>
    </div>
    """, unsafe_allow_html=True)
