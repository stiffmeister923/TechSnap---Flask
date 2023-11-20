from flask import Flask, request, session, jsonify, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from prebuiltsystem import run_function
import requests
import os
from haystack.pipelines.standard_pipelines import TextIndexingPipeline,ExtractiveQAPipeline
from haystack.nodes import BM25Retriever, FARMReader
from haystack.document_stores.memory import InMemoryDocumentStore

app = Flask(__name__)
app.secret_key = 'Matt'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'  # SQLite database file
db = SQLAlchemy(app)
class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    subject = db.Column(db.String(200))
    message = db.Column(db.Text, nullable=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/your_route')
def your_route():
    try:
        # Simulate data retrieval or processing
        result = {"prediction": "Your prediction", "image_url": "your_image_url.jpg"}
        session['resultImg'] = result
        # Render the 'new.html' template with the simulated data    
        return render_template('new.html', result=result)

    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/process_form', methods=['POST'])
def process_form():
    if request.method == 'POST':
        # Get user selections from the form
        category = request.form['category']
        subcategory = request.form['subcategory']
        budget = int(request.form['budget'])
        results =run_function(category, subcategory, budget)
        unicodess = u"\u279C"
        # Process the selections (you can replace this with your actual processing logic)
        result = session.get('resultImg', {})
        category = f'{category} {unicodess} {subcategory}'
        inference = session.get('inference',{})
        #template results[selected,best][per computer][attributes]
        # Pass the result to the template
        return render_template('recommend.html', recommend1=results[0], recommend2 =results[1] , category=category.upper(), result=result, result1=inference)
    

@app.route('/detect', methods=['POST'])
def detect_objects():
    try:
        # Get the image file from the user's request
        image = request.files['fileUpload']

        if not image:
            raise ValueError("No Image Provided")

        # Prepare the data to send to the Ultralytics API
        url = "https://api.ultralytics.com/v1/predict/ynSBfJYksFVQUmJiKkmu"
        headers = {"x-api-key": "1f752d22c016d6b879ff47610a64828ef2d080d29e"}
        data = {"size": 640, "confidence": 0.25, "iou": 0.45}

        # Make a request to the Ultralytics API
        response = requests.post(url, headers=headers, data=data, files={"image": image})

        # Check for a successful response
        response.raise_for_status()
        session['inference'] = response.json()
        # Render the 'new.html' template with the API response
        return render_template('new.html', result1=response.json())

    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    try:
        # Get form data from the request
        full_name = request.form.get('full-name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')

        # Save form data to the database
        new_message = ContactMessage(full_name=full_name, email=email, subject=subject, message=message)
        db.session.add(new_message)
        db.session.commit()

        return redirect(url_for('index'))

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/answer', methods=['POST'])
def answer():
    if request.method == 'POST':
        bro = "contextcopy/your_context"
        model = "stiffmeister923/BERT-trained-computerparts"
        document_store = InMemoryDocumentStore(use_bm25=True)
        files_to_index = [bro + "/" + f for f in os.listdir(bro)]
        indexing_pipeline = TextIndexingPipeline(document_store)
        indexing_pipeline.run_batch(file_paths=files_to_index)
        retriever = BM25Retriever(document_store=document_store)
        reader = FARMReader(model_name_or_path=model, use_gpu=True)
        question = request.form['question']
        pipe = ExtractiveQAPipeline(reader, retriever)
        prediction = pipe.run(
        query=question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
        )
        # Call your Q&A model to get the answer based on the user's question
        # Replace the following line with your actual Q&A model inference code
       

        return render_template('index.html', question=question, answer=prediction['answers'][0].answer)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
