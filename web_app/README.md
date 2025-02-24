# Flask Web Application for Image Generation

This Flask-based web application allows users to upload an image, select a model, checkpoint, and resolution, and define a prompt to generate a new image. The input image is resized symmetrically to the specified resolution. Only PNG images are accepted.

This tool is helpful for evaluating the behavior of generative models under different conditions.

---

## Usage

### 1. Create a Virtual Environment

Create a virtual environment to isolate dependencies:

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:
  ```bash
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### 2. Install Requirements

Install the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

Start the Flask application:

```bash
python app.py
```

The application will be exposed on port `5000`. Open your browser and navigate to `http://127.0.0.1:5000/` to access the web interface.

---

## Notes

- This application is intended to run on the same server where the OmniGen or other models are configured.
- Ensure the model is in the correct location as specified in the `generate_image.py` file.
- Only PNG images are accepted as input.

---

For more details, refer to the `generate_image.py` file and ensure the model is properly set up before running the application.