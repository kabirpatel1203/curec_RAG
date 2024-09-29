# app.py
from flask import Flask, render_template, request
from main import MedicalTermProcessor

app = Flask(__name__)
processor = MedicalTermProcessor()

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        text = request.form['medical_text']
        results = processor.process_text(text)
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)

