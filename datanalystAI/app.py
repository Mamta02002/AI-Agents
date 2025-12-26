from flask import Flask, render_template, request, url_for, redirect, jsonify, send_from_directory
from agent.analysis import analyze_data
import numpy as np
import pickle
import os
from pathlib import Path

app = Flask(__name__)

# Route to serve the index.html page
@app.route('/')
def index():
    return render_template('agent_index.html')

@app.route('/exports/<path:filename>')
def serve_export_file(filename):
    return send_from_directory('exports', filename)

@app.route('/output/<path:filename>')
def serve_output_file(filename):
    return send_from_directory('output', filename)

# Route to handle analysis
@app.route('/analyze', methods=['POST'])
def start_analyzing():
    # Get uploaded file and form data
    uploaded_file = request.files.get('data')
    description = request.form.get('description')
    n = request.form.get('n')

    # Save uploaded file temporarily
    if uploaded_file:
        file_path = os.path.join('uploads', uploaded_file.filename)
        uploaded_file.save(file_path)
    else:
        return jsonify({'error': 'No file uploaded'}), 400

    # Call your model function (replace with your actual logic)
    html_path, zip_path = analyze_data(file_path=file_path, data_description=description, n=n)
    # html_path = os.path.join(Path(__file__).parent, 'output', 'analysis_report.html')
    # Read the HTML content
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Return HTML content as JSON
    return jsonify({'html': html_content})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    app.run(debug=True)
