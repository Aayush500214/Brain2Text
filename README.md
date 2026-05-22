<div align="center">
  <img src="https://img.shields.io/badge/Brain_to_Text-2025-blue?style=for-the-badge&logo=brain" alt="Logo" />
  
  <h1>🧠 Brain-to-Text '25</h1>
  
  <p>
    <strong>A State-of-the-Art Production Demo Application for Brain-to-Text Inference</strong>
  </p>

  <p>
    <a href="#-features">Features</a> •
    <a href="#-repository-contents">Files</a> •
    <a href="#-getting-started">Getting Started</a> •
    <a href="#-fallback-demo-mode">Demo</a>
  </p>
  
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white" />
  <img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white" />
</div>

<br/>

## ✨ Overview

Welcome to the **Brain-to-Text '25** project repository! This repository houses a production-quality frontend and a robust, Colab-ready backend API designed to seamlessly demonstrate neural decoding and brain-to-text generation.

---

## 📂 Repository Contents

| File / Folder | Description |
| :--- | :--- |
| 📄 `index.html` | The complete, single-file frontend. **No build tools required**—just double-click to open in any modern browser! |
| 🐍 `colab_app.py` | The Flask backend API tailored specifically for running on Google Colab. |
| 🚀 `colab_setup.py` | The deployment cell you run inside Google Colab to start the API and expose it publicly via ngrok. |
| 📄 `brain2Text Merged.pdf` | **Project Documentation**: The complete, merged research and project documentation. |

---

## 🚀 Getting Started

Follow these steps to run the live demo and experience real-time inference.

### 1️⃣ Start the Backend (Google Colab)

1. Open a new **Google Colab** notebook.
2. Upload the following files to the Colab file explorer:
   - `colab_app.py`
   - `phoneme_to_text.py` *(if using an LLM for text generation)*
   - Your trained model weights (e.g., `best_model_lstm_pretrained.pth` or `.pkl`).
3. Copy the contents of `colab_setup.py` into a notebook cell.
4. Replace `"YOUR_NGROK_AUTH_TOKEN_HERE"` in the cell with your free ngrok token from [dashboard.ngrok.com](https://dashboard.ngrok.com/).
5. Run the cell and wait for the success message:  
   `🚀 API IS LIVE AT: https://[random-string].ngrok.app`  
   *(Copy this URL for the next step!)*

### 2️⃣ Launch the Frontend

1. Double-click `index.html` on your local computer to open it in your preferred browser (Chrome, Edge, Firefox).
2. Click on **"LIVE API MODE"** at the top right of the interface.
3. Paste the **ngrok URL** you copied earlier into the input box.
4. Click the upload button to select an extracted `.npy` trial file.
5. Hit **"RUN INFERENCE"** and watch the magic happen! ✨

---

## 🎭 Fallback: Demo Mode

If the backend is down, or you find yourself without an internet connection, don't worry! 

Simply click **"DEMO MODE"**. This runs completely locally in your browser with pre-loaded trial data, simulating the API response perfectly for a seamless presentation.

---

<div align="center">
  <p>Built with ❤️ for Brain-to-Text '25</p>
</div>
