from flask import Flask, render_template, request, redirect, url_for, jsonify
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


@app.route('/')
def index():
    """Render the main page."""
    # Get processed images to display
    processed_images = [f for f in os.listdir(app.config['PROCESSED_FOLDER']) if f.startswith('inference')]
    return render_template('index.html', processed_images=processed_images)

# Define a route for the root URL ("/")
@app.route('/upload', methods=['POST'])
def upload():
    
    """Handle file upload and processing asynchronously."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    # If the file is valid
    if file and allowed_file(file.filename):
        # clear previous pictures 
        clear_all_in_folder()
        file_extension=file.filename.split(".")[1]

        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], "test_image."+file_extension)
        file.save(filepath)            

        # Get the text inputs from the form
        model = request.form.get('model')
        checkpoint = request.form.get('checkpoint')
        instruction = request.form.get('instruction')
        resolution = int(request.form.get('resolution'))

        #modify the extension and resolution 
        resize_and_save_image(filepath,(resolution,resolution))

        print(model,checkpoint,instruction)
        generate_multiple(model,checkpoint,instruction)
        processed_images = [f for f in os.listdir(app.config['PROCESSED_FOLDER']) if f.startswith('inference')]
        print(processed_images)
        return jsonify({'processed_images': processed_images})
    return jsonify({'error': 'Invalid file type'}), 400


# Run the application on port 5000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)