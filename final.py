import os
import json
import requests
import streamlit as st
import pandas as pd
import PyPDF2

# ------------------- Hugging Face API Config -------------------
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # token stored as secret in Spaces
HF_MODEL = "tiiuae/falcon-7b-instruct"         # replace with your model
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# ------------------- Helper Functions -------------------
def call_hf(prompt, max_new_tokens=256):
    """Call Hugging Face Inference API."""
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.3},
        "options": {"wait_for_model": True}
    }
    r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
    r.raise_for_status()
    out = r.json()
    if isinstance(out, list) and "generated_text" in out[0]:
        return out[0]["generated_text"]
    return str(out)

def parse_json_output(raw_text):
    """Convert raw string from model to JSON."""
    try:
        return json.loads(raw_text.strip())
    except:
        return [{"raw": raw_text}]

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="AI-Powered Exam Generator", layout="wide")
st.title("📘 AI-Powered Exam Question Generator")

st.markdown("""
Upload your **class notes** and let AI generate:
- 📌 Multiple Choice Questions (MCQs)
- ✍️ Short Answer Questions with Model Answers

Dataset base: MIT OpenCourseWare (OCW) text data.
""")

# ------------------- File Upload -------------------
uploaded_file = st.file_uploader("Upload your notes (.txt or .pdf)", type=["txt", "pdf"])

if uploaded_file is not None:
    file_text = ""

    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                file_text += text + "\n"
    else:
        file_text = uploaded_file.read().decode("utf-8", errors="ignore")

    st.subheader("📄 Preview of Uploaded Notes")
    st.text_area("Notes Content", file_text[:2000], height=200)

    # ------------------- Question Type -------------------
    q_type = st.radio("Select Question Type", ["MCQs", "Short Answers"])

    if st.button("✨ Generate Questions"):
        with st.spinner("Generating... ⏳"):
            # Construct prompt for Hugging Face API
            if q_type == "MCQs":
                prompt = f"""
                You are an exam writer. From the passage below, generate 3 MCQs in JSON format.
                Passage: <<< {file_text[:1500]} >>>
                Format:
                [
                  {{
                    "question": "...",
                    "options": {{"A":"...","B":"...","C":"...","D":"..."}} ,
                    "answer": "...",
                    "explanation": "..."
                  }}
                ]
                """
            else:  # Short Answers
                prompt = f"""
                You are an exam writer. From the passage below, generate 3 short-answer questions in JSON format.
                Passage: <<< {file_text[:1500]} >>>
                Format:
                [
                  {{
                    "question": "...",
                    "model_answer": "...",
                    "rubric": "0/1/2 scoring"
                  }}
                ]
                """

            raw_output = call_hf(prompt)
            items = parse_json_output(raw_output)

            df = pd.DataFrame(items)
            st.success("✅ Questions Generated!")
            st.dataframe(df, use_container_width=True)

            # Download CSV
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", data=csv, file_name="questions.csv", mime="text/csv")
