# PDF Answering AI

## Overview

This project aims to develop a web-based application that uses natural language processing (NLP) techniques to provide quick and accurate responses to user queries based on the content of an uploaded PDF document. The application leverages the `distilbert-base-uncased-distilled-squad` model from the Hugging Face Transformers library for question-answering tasks.

## Objectives

- **Seamless PDF Text Extraction:** Efficiently extract text from uploaded PDF files while retaining structure and context.
- **Accurate Question Answering:** Use a pre-trained NLP model (DistilBERT) to interpret and answer questions based on the extracted text from the PDF.
- **User-Friendly Web Interface:** Develop a web interface using Flask that allows users to upload PDF files and ask questions.
- **Maintainability and Scalability:** Structure the project for easy maintenance and potential future enhancements.
- **Compatibility and Security:** Ensure the application is compatible across different operating systems and web browsers, and handle file uploads securely.

## Components

### 1. `bert_model.py`

This script sets up the BERT model for question answering using the Hugging Face Transformers library.

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

class BERTQA:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
        self.model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

    def get_answer(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
        return answer

bert_qa = BERTQA()


### app.py

The `app.py` file sets up the Flask web application and handles the routes for file upload and question answering.

- **Imports:** 
  - `Flask`, `request`, `render_template`, `redirect`, and `url_for` are imported from the Flask library.
  - `extract_text_from_pdf` is imported from `pdf_extractor.py` to extract text from the uploaded PDF.
  - `bert_qa` is imported from `bert_model.py` to handle question-answering tasks.

- **Flask Application:**
  - An instance of the Flask application is created with `app = Flask(__name__)`.
  - `context` is a global variable to store the extracted text from the uploaded PDF.

- **Routes:**
  - `/`: The home route renders `index.html`, which is the file upload form.
  - `/upload`: Handles the PDF file upload, saves the uploaded file, extracts text from it, stores it in the `context` variable, and redirects to the `/ask` route.
  - `/ask`: Handles the question-answering functionality. If a POST request is received (a question is submitted), it uses the `bert_qa` model to get an answer from the `context` and renders `ask.html` with the response.

- **Running the Application:** The application runs in debug mode with `app.run(debug=True)`.

```python
from flask import Flask, request, render_template, redirect, url_for
from pdf_extractor import extract_text_from_pdf
from bert_model import bert_qa

app = Flask(__name__)

context = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global context
    file = request.files['file']
    file.save('uploaded_file.pdf')
    context = extract_text_from_pdf('uploaded_file.pdf')
    return redirect(url_for('ask'))

@app.route('/ask', methods=['GET', 'POST'])
def ask():
    response = ""
    if request.method == 'POST':
        question = request.form['question']
        response = bert_qa.get_answer(question, context)
    return render_template('ask.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)
