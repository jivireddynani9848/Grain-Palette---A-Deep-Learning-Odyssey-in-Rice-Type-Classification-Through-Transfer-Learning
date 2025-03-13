from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub 
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = tf.keras.models.load_model('my_model.keras', custom_objects={'KerasLayer': hub.KerasLayer})
labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img = cv2.imread(filepath)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        prediction = model.predict(img)
        predicted_label = labels[np.argmax(prediction)]
        
        return render_template('result.html', prediction=predicted_label, image_url=filepath)
    
    return '''
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" required>
        <input type="submit" value="Upload and Classify">
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
