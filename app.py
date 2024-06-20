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
