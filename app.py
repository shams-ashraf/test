import streamlit as st
from PIL import Image
import fitz, io
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import torch
import os

st.title("🧾 PDF Image OCR with Qwen2-VL")

# إعداد المجلد
OUTPUT_FOLDER = "extracted_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# إعداد النموذج
model_name = "prithivMLmods/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

prompt_text = (
    "Please carefully read the image and extract all text clearly and completely. "
    "Organize it in a structured, human-readable way (titles, paragraphs, lists). "
    "Ignore watermarks or irrelevant marks."
)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    MIN_WIDTH, MIN_HEIGHT = 40, 40
    image_counter = 1

    for page_index, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                continue

            image_name = f"image_{image_counter}.png"
            image_path = os.path.join(OUTPUT_FOLDER, image_name)
            image.save(image_path)

            messages = [{"role": "user", "content":[{"type":"image"}, {"type":"text", "text":prompt_text}]}]
            inputs_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=[inputs_text], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=1024)
                output_text = processor.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

            st.subheader(f"Image {image_counter} (Page {page_index+1})")
            st.image(image, caption=image_name)
            st.text_area("Extracted Text", output_text, height=200)

            image_counter += 1
