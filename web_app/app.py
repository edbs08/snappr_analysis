from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
from scripts.generate_images import *

# Create a Flask application
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/'
PROCESSED_FOLDER = 'static/'
ALLOWED_EXTENSIONS = {'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure upload and processed directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# Define a route for the root URL ("/")
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If no file is selected
        if file.filename == '':
            return redirect(request.url)
        
        # If the file is valid
        if file and allowed_file(file.filename):
            # Save the uploaded file
            #********** missing preprocess for inference
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Get the selected model
            selected_model = request.form.get('model')  # Get the selected model from the radio buttons
            print(selected_model)
            generate_multiple(selected_model)

    # Get processed images to display
    processed_images = [f for f in os.listdir(app.config['PROCESSED_FOLDER']) if f.startswith('inference')]
    
    return render_template('index.html', processed_images=processed_images)

# Run the application on port 5000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)