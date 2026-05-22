"""
Run this entire cell in Google Colab to start the backend API.
It will install dependencies, start the Flask app in the background, 
and expose it via ngrok.
"""

import os
import subprocess
import time
import urllib.request
import json

# 1. Install dependencies
print("Installing dependencies...")
subprocess.run(["pip", "install", "-q", "flask", "flask-cors", "pyngrok", "g2p_en", "requests"], check=True)

from pyngrok import ngrok

# 2. Add your ngrok token here
# Get it from https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN_HERE" 

if NGROK_AUTH_TOKEN != "YOUR_NGROK_AUTH_TOKEN_HERE":
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
else:
    print("\nWARNING: Please set your NGROK_AUTH_TOKEN in the script to expose the API publicly.\n")

# 3. Create the Flask app file if it doesn't exist in Colab
app_code = """
import os
import io
import torch
import torch.nn as nn
import numpy as np
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# [Paste the contents of colab_app.py here or ensure it's uploaded to Colab workspace]
# For the sake of the setup script, ensure colab_app.py is uploaded alongside this.
"""

if not os.path.exists("colab_app.py"):
    print("\nERROR: colab_app.py not found in the current directory.")
    print("Please upload colab_app.py and your model weights (.pth or .pkl) to the Colab files area.")
else:
    print("Found colab_app.py.")

# 4. Expose the port with ngrok
try:
    public_url = ngrok.connect(5000)
    print("\n" + "="*50)
    print(f"🚀 API IS LIVE AT: {public_url.public_url}")
    print("Copy this URL and paste it into the 'ngrok API URL' box in the frontend.")
    print("="*50 + "\n")
    
    # 5. Start Flask in the background
    print("Starting Flask server...")
    flask_process = subprocess.Popen(["python", "colab_app.py"])
    
    # Keep the cell running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        flask_process.terminate()
        ngrok.kill()
        
except Exception as e:
    print(f"Failed to start ngrok/Flask: {e}")
