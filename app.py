import os
import re
import io
import fitz  # PyMuPDF
import docx
from PIL import Image
import pytesseract
import uuid
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# OCR setup
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Settings
DOCUMENTS_FOLDER = "./documents"
OUTPUT_FOLDER = "extracted_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Text cleaning ---
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def structure_text_into_paragraphs(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    paragraphs = []
    current_paragraph = []
    for i, line in enumerate(lines):
        if len(line.split()) < 3 and not (line[0].isupper() or re.match(r'^[\d]+[\.\):]', line)):
            continue
        is_heading = (line.isupper() and len(line.split()) <= 10) or (len(line.split()) <= 6 and line[0].isupper() and line.endswith(':'))
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
        if i < len(lines)-1:
            next_line = lines[i+1]
            next_words = next_line.split()
            next_is_new_section = re.match(r'^[\d]+[\.\)]\s', next_line) or re.match(r'^[•\-\*]\s', next_line) or (len(next_words)<=6 and next_line[0].isupper()) or next_line.isupper()
        if ends_with_punctuation or next_is_new_section or i==len(lines)-1:
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
    return structured_text.strip()

# --- OCR from images ---
def extract_and_structure_text_from_image(image):
    raw_text = pytesseract.image_to_string(image, lang='eng+ara+deu')
    structured_text = structure_text_into_paragraphs(raw_text)
    if '|' in structured_text or '\t' in structured_text or re.search(r'\d+\s+\w+\s+\d+', structured_text):
        structured_text = "📊 [Table content from image]\n\n" + structured_text
    return structured_text

# --- Extract tables from DOCX ---
def extract_table_from_docx(table, table_number=None):
    if len(table.rows)==0: return ""
    headers = [clean_text(cell.text) for cell in table.rows[0].cells if cell.text.strip()]
    if not headers: return ""
    lines = []
    title = f"Table {table_number}" if table_number else "Table"
    lines.append(f"\n┌{'─'*58}┐\n│  📊 {title}{' '*(54-len(title))}│\n└{'─'*58}┘\n")
    lines.append("📋 Columns:")
    for idx, h in enumerate(headers,1):
        lines.append(f"  {idx}. {h}")
    lines.append(f"\n{'─'*60}\n📊 Data:\n")
    for i,row in enumerate(table.rows[1:],1):
        cells = [clean_text(c.text) for c in row.cells]
        if not any(cells): continue
        lines.append(f"▸ Row {i}:")
        for h,v in zip(headers,cells):
            lines.append(f"  • {h}: {v if v else '[Empty]'}")
        lines.append("")
    lines.append(f"{'─'*60}\n📈 Summary: {len(table.rows)-1} rows, {len(headers)} columns\n{'─'*60}\n")
    return "\n".join(lines)

# --- Extract text from DOCX ---
def extract_docx_text(file_path):
    doc = docx.Document(file_path)
    all_text = []
    table_count = 0
    for block in doc.element.body:
        if block.tag.endswith('tbl'):
            table_count += 1
            table = doc.tables[table_count-1]
            table_text = extract_table_from_docx(table, table_number=table_count)
            all_text.append({"text": table_text, "type": "table", "source": os.path.basename(file_path)})
        elif block.tag.endswith('p'):
            idx = len(all_text)
            para_text = doc.paragraphs[idx].text
            if para_text.strip():
                all_text.append({"text": structure_text_into_paragraphs(para_text), "type":"paragraph", "source": os.path.basename(file_path)})
    return all_text, table_count

# --- Extract text from PDF ---
def extract_pdf_text(file_path):
    doc = fitz.open(file_path)
    all_text = []
    table_count = 0
    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            image_name = os.path.join(OUTPUT_FOLDER,f"{uuid.uuid4().hex}.png")
            image.save(image_name)
            img_text = extract_and_structure_text_from_image(image)
            all_text.append({"text": img_text, "type":"image", "source": os.path.basename(file_path), "page": page_num+1})
        page_text = page.get_text()
        if page_text.strip():
            all_text.append({"text": structure_text_into_paragraphs(page_text), "type":"paragraph", "source": os.path.basename(file_path), "page": page_num+1})
    return all_text, table_count

# --- Extract text from TXT ---
def extract_txt_text(file_path):
    with open(file_path,"r",encoding="utf-8") as f:
        content = f.read()
    return [{"text": structure_text_into_paragraphs(content), "type":"paragraph", "source": os.path.basename(file_path)}], 0

# --- Smart chunking ---
def create_smart_chunks(text_entries, chunk_size=700, overlap=200):
    chunks = []
    for entry in text_entries:
        words = entry["text"].split()
        if len(words) <= chunk_size:
            if entry["text"].strip(): 
                chunks.append({**entry, "chunk": entry["text"]})
            continue
        for i in range(0,len(words), chunk_size-overlap):
            chunk_text = " ".join(words[i:i+chunk_size])
            if len(chunk_text.split())>=30:
                new_entry = entry.copy()
                new_entry["chunk"] = chunk_text
                chunks.append(new_entry)
    return chunks

# --- Embed chunks in ChromaDB ---
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
    for i in range(0,len(chunks),batch_size):
        batch = chunks[i:i+batch_size]
        collection.add(
            documents=[c["chunk"] for c in batch],
            ids=[f"chunk_{i+j}" for j in range(len(batch))],
            metadatas=[{"source": c.get("source",""), "page": c.get("page",""), "type": c.get("type","")} for c in batch]
        )
    return collection

# --- Main processing ---
all_files = [os.path.join(DOCUMENTS_FOLDER,f) for f in os.listdir(DOCUMENTS_FOLDER)]
all_entries = []
for file_path in all_files:
    print(f"\nProcessing: {os.path.basename(file_path)}")
    print("="*70)
    entries, tables_count = process_document(file_path)
    print(f"\n✅ Extracted {len(entries)} entries")
    print(f"📊 Detected {tables_count} tables\n")
    all_entries.extend(entries)

all_chunks = create_smart_chunks(all_entries)
collection = embed_chunks(all_chunks)
print(f"\n✅ {len(all_chunks)} chunks embedded and stored in ChromaDB.")
