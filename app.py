import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

st.set_page_config(page_title="AI Story Generator", layout="centered")

st.title("AI Story Generator from Prompts")
st.write("Enter a creative prompt and let the AI write a short story for you!")

@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

prompt = st.text_area("Enter your story prompt:", height=150)

max_length = st.slider("Story length (in tokens):", min_value=50, max_value=300, value=150)

if st.button("Generate Story"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt first.")
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        story = tokenizer.decode(output[0], skip_special_tokens=True)
        st.subheader("🧠 Generated Story")
        st.write(story)

        