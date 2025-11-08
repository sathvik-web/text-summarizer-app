import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

st.set_option('client.showErrorDetails', True)

st.title("Abstractive Text Summarizer (T5 Transformer)")
st.write(
    "This web app uses the **T5-small** model from Hugging Face to generate concise summaries "
    "of longer text. Enter any paragraph below and click **Summarize**!"
)

@st.cache_resource
def load_model():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

user_input = st.text_area("Enter text to summarize:", height=250, placeholder="Paste or type your paragraph here...")

if st.button("Summarize"):
    if user_input.strip():
        st.info("⏳ Generating summary... please wait.")
        input_text = "summarize: " + user_input
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        summary_ids = model.generate(
            inputs,
            max_length=150,
            min_length=40,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.success("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text first.")
