# Paraphraser App (Hugging Face and Streamlit)

### CLICK TO WATCH THE VIDEO
[![Build Your Own Paraphrase App in 7 Minutes!](https://img.youtube.com/vi/56upVPEJbm0/0.jpg)](https://youtu.be/56upVPEJbm0)


This repository contains the core Python code for a paraphrase web application built using the Hugging Face Transformers library.

- This script leverages the humarin/chatgpt_paraphraser_on_T5_base model to generate paraphrases of input text.
- Users can provide text for paraphrasing, and the script will return a single paraphrased version.


Installation:
You can install these libraries using pip:
```
pip install transformers nltk streamlit
```
You may also need to download additional NLTK data for tokenization. You can do this using the following command:
```
import nltk
nltk.download('punkt_tab')
```

To run this app, write this on your command line:
```
streamlit run web_app.py
```

