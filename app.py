from flask import Flask, request, session, jsonify, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from prebuiltsystem import run_function
import pinecone
from sentence_transformers import SentenceTransformer
from component import ramfunc, cpufunc,gpufunc,psufunc,mobofunc,storfunc
import requests
import os

 
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

pinecone.init(
	api_key='d5be10cf-4dd1-49ad-aac7-789a7c3827e9',
	environment='gcp-starter'
)
index_name = "techsnap"

index = pinecone.Index(index_name)
retriever = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
def get_context(question, top_k=3):
    # generate embeddings for the question
    xq = retriever.encode([question]).tolist()
    # search pinecone index for context passage with the answer
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    # extract the context passage from pinecone search result
    c = [x["metadata"]['context'] for x in xc["matches"]]
    return c

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/component')
def findComponents():
    return render_template('components.html')    

@app.route('/pre_built')
def findPre_built():
    return render_template('recommend.html') 
   
@app.route('/querying')
def querying():
    return render_template('questionanswering.html')    
     

 
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
        
        inference = session.get('inference',{})
        category = f'{category} {unicodess} {subcategory}'
        #template results[selected,best][per computer][attributes]
        # Pass the result to the template
        return render_template('recommend.html', recommend1=results[0], recommend2 =results[1] , category=category.upper())
     

@app.route('/detect', methods=['POST'])
def detect_objects():
    try:
        # Get the image file from the user's request
        image = request.files['fileUpload']
        #postProcessed = clahe(image)
        #print(postProcessed)
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
        
        question = request.form['question']
        prediction = get_context(question)
        

        return render_template('questionanswering.html', question=question, answer=prediction)
    
@app.route('/goto')
def getComponentHTML():
    result = session.get('resultImg', {})
    inference = session.get('inference',{})
    return render_template('component.html',result=result, result1=inference)

@app.route('/getcomponent', methods=['GET', 'POST'])
def componentFunction():
    result = session.get('resultImg', {})
    inference = session.get('inference',{})
    if request.method == 'POST':
        selectedoption = request.form['component']
        budget = float(request.form['budget'])
        category = request.form['category']
        if category == 'ram':
            data = ramfunc(budget, selectedoption)
        elif category == 'cpu':
            data = cpufunc(budget, selectedoption)
        elif category == 'gpu':
            data = gpufunc(budget, selectedoption)
        elif category == 'psu':
            data = psufunc(budget, selectedoption)
        elif category == 'mobo':
            data = mobofunc(budget, selectedoption)
        elif category == 'storage':
            data = storfunc(budget, selectedoption)
        attributes = set()
        for item in data:
            attributes.update(item.keys())

    # Convert set to list and sort for consistent order
    attributes = sorted(list(attributes))
    return render_template('component.html',result=result, result1=inference,attributes=attributes,data=data)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host="0.0.0.0")
