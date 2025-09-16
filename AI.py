import os
import json
import requests
import streamlit as st
import pandas as pd

# ------------------- Config -------------------
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # set in your env
HF_MODEL = "gpt2"  # replace with better instruct model like "tiiuae/falcon-7b-instruct"
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}


# ------------------- Helper Functions -------------------
def call_hf(prompt, max_new_tokens=256):
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
    try:
        return json.loads(raw_text.strip())
    except:
        return [{"raw": raw_text}]


# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="AI Exam Generator", layout="wide")
st.title("📘 AI-Powered Exam Question Generator")

st.markdown(
    """
    Upload your **class notes** and let AI generate:
    - 📌 Multiple Choice Questions (MCQs)
    - ✍️ Short Answer Questions with Model Answers

    Dataset base: MIT OpenCourseWare (OCW) text data.
    """
)

uploaded = st.file_uploader("Upload your notes (.txt or .pdf)", type=["txt", "pdf"])
q_type = st.radio("Select Question Type", ["MCQs", "Short Answers"])

if uploaded:
    text = uploaded.read().decode("utf-8", errors="ignore")

    if st.button("✨ Generate Questions"):
        with st.spinner("Generating... ⏳"):
            if q_type == "MCQs":
                prompt = f"""
                You are an exam writer. From the passage below, generate 3 MCQs in JSON format.
                Passage: <<< {text[:1500]} >>>
                Format:
                [
                  {{
                    "question": "...",
                    "options": {{"A":"...","B":"...","C":"...","D":"..."}},
                    "answer": "B",
                    "explanation": "..."
                  }}
                ]
                """
            else:
                prompt = f"""
                You are an exam writer. From the passage below, generate 3 short-answer questions in JSON.
                Passage: <<< {text[:1500]} >>>
                Format:
                [
                  {{
                    "question": "...",
                    "model_answer": "...",
                    "rubric": "0/1/2 scoring"
                  }}
                ]
                """

            raw = call_hf(prompt)
            items = parse_json_output(raw)

        # Convert to DataFrame for display
        df = pd.DataFrame(items)
        st.success("✅ Questions Generated!")

        # Pretty, sortable table
        st.dataframe(df, use_container_width=True)

        # Download as CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv, file_name="questions.csv", mime="text/csv")
