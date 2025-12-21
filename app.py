import streamlit as st
import fitz
from PIL import Image
import io
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import os

st.title("🧾 PDF OCR with Qwen2‑VL‑2B‑OCR")

# رفع موديل Qwen2‑VL‑2B‑OCR (public)
@st.cache_resource
def load_model():
    model_name = "JackChew/Qwen2‑VL‑2B‑OCR"
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return processor, model

processor, model = load_model()
device = next(model.parameters()).device

uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")
if uploaded_pdf:
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    output_folder = "extracted_images"
    os.makedirs(output_folder, exist_ok=True)

    st.success(f"Loaded PDF — {len(doc)} pages")

    image_counter = 1
    for page_index, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))

            # حفظ الصورة
            image_name = f"page{page_index+1}_img{image_counter}.png"
            image_path = os.path.join(output_folder, image_name)
            image.save(image_path)

            # تجهيز الإدخال للنموذج
            prompt = "Extract all text clearly and completely from this image."
            messages = [
                {"role": "user", "content":[{"type":"image"}, {"type":"text", "text": prompt}]}
            ]
            inputs_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=[inputs_text], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # inference
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=1500)
            extracted_text = processor.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

            st.subheader(f"Page {page_index+1} — Image {image_counter}")
            st.image(image, caption=image_name)
            st.text_area("Extracted text", extracted_text, height=200)

            image_counter += 1
