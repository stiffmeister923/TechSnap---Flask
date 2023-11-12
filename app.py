from flask import Flask, request, jsonify, render_template

import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/your_route')
def your_route():
    try:
        # Simulate data retrieval or processing
        result = {"prediction": "Your prediction", "image_url": "your_image_url.jpg"}

        # Render the 'new.html' template with the simulated data
        return render_template('new.html', result=result)

    except Exception as e:
        return jsonify({'error': str(e)})

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

        # Render the 'new.html' template with the API response
        return render_template('new.html', result=response.json())

    except Exception as e:
        return jsonify({'error': str(e)})
    


    
    

if __name__ == '__main__':
    app.run(debug=True)
