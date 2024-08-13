from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory

from werkzeug.utils import secure_filename
import os
from model import predict_image, load_vision_transformer_model

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_PATH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the Vision Transformer model
model = load_vision_transformer_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Perform prediction
        prediction = predict_image(filepath, model)
        class_labels = {0: 'Grassy', 1: 'Marssy', 2: 'Rocky', 3: 'Sandy', 4: 'Invalid Image'}
        
        return render_template('result.html', prediction=class_labels[prediction], image_url=url_for('uploaded_file', filename=filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
