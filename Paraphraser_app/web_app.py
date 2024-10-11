from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from nltk.tokenize import sent_tokenize
import streamlit as st
import time


device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

def paraphrase_one_sentence(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=1,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)
    
    # with autocast():  # Enable mixed precision
    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res


def paraphrase(paragraph):
    # Tokenize the paragraph into sentences
    sentences = sent_tokenize(paragraph)

    paraphrased_sentences = []

    for sentence in sentences:
        paraphrased_result = paraphrase_one_sentence(sentence, num_beams=2, num_beam_groups=2, num_return_sequences=1, max_length=128)
        paraphrased_sentences.append(paraphrased_result[0])  # Take the first paraphrased output

    # Join the paraphrased sentences back together
    paraphrased_paragraph = " ".join(paraphrased_sentences)
    return paraphrased_paragraph

# Streamlit app layout
# Style and Layout: Set the background and header colors for a blue hue aesthetic
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;  /* Alice Blue */
        }
        h1, h2, h3 {
            color: #007acc;  /* Soft Blue for titles */
        }
        .reportview-container {
            background: #f0f8ff;
        }
        textarea {
            background-color: #e0f7fa;
        }
        .stButton>button {
            background-color: #007acc;
            color: white;
        }
        .stTextArea>div>div>textarea {
            color: #007acc;
        }
    </style>
    """, unsafe_allow_html=True)
# Add the banner image at the top
st.image("mohan youtube banner.png", use_column_width=True)


st.title("Paraphrase App")
st.markdown("Enter your text on the left, and see the paraphrased version on the right!")

# Split the page into two columns
col1, col2 = st.columns(2)

with col1:
    st.header("Original Text")
    input_text = st.text_area("Enter your text here:", height=500)

with col2:
    st.header("Paraphrased Text")
    if st.button("Paraphrase"):
        if input_text:
            # Start timer
            start_time = time.time()
            
            # Paraphrase the input text
            paraphrased_text = paraphrase(input_text)
           
            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            st.write(paraphrased_text)
            st.write("---------")
            st.write(f"Time taken for inference: {elapsed_time:.2f} seconds")
        else:
            st.write("Please enter text in the left column.")


