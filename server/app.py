"""
app.py — Brain2Text Flask API server.

Endpoints:
    GET  /api/health               Server status + model info
    POST /api/decode               JSON neural_data array → phonemes + text
    POST /api/predict              Multipart .npy file → phonemes + text   (frontend "live mode")
    POST /api/phonemes-to-text     Raw phoneme string → English text
    GET  /api/demo                 Batch demo with sample phonemes
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

import numpy as np
import os
import io

load_dotenv()

from inference import load_model, run_inference
from phoneme_to_text import phonemes_to_text, batch_phonemes_to_text

app = Flask(__name__)
CORS(app)

# ─── Lazy model cache ────────────────────────────────────────────────────────

_model = None
_meta = None


def get_model():
    global _model, _meta
    if _model is None:
        _model, _meta = load_model()
    return _model, _meta


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    has_api_key = bool(os.environ.get("POLLINATION_API_KEY"))
    try:
        m, meta = get_model()
        model_info = {
            "loaded": True,
            "arch": meta["arch"]["model_type"],
            "layers": meta["arch"]["num_layers"],
            "hidden": meta["arch"]["hidden_size"],
            "best_per": round(meta["best_per"], 4) if meta["best_per"] is not None else None,
        }
    except Exception as e:
        model_info = {"loaded": False, "error": str(e)}

    return jsonify({
        "status": "ok",
        "model": model_info,
        "api_key_configured": has_api_key,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Live-mode endpoint used by the frontend.

    Accepts:
        multipart/form-data  →  file: .npy   (shape [T, 512])
        OR
        application/json     →  { "neural_data": [[...], ...] }

    Returns:
        {
            "phonemes": "DH AH | K AE T ...",
            "text": "The cat...",
            "inference_time_ms": 134.5,
            "sequence_length": 120,
            "per": 0.0,
            "success": true
        }
    """
    try:
        # ── Parse input ──────────────────────────────────────────────────────
        if request.content_type and "multipart" in request.content_type:
            if "file" not in request.files:
                return jsonify({"error": "No file uploaded", "success": False}), 400
            f = request.files["file"]
            raw = f.read()
            neural_data = np.load(io.BytesIO(raw), allow_pickle=False)
        else:
            data = request.get_json()
            if not data or "neural_data" not in data:
                return jsonify({"error": "Missing 'neural_data'", "success": False}), 400
            neural_data = np.array(data["neural_data"], dtype=np.float32)

        # ── Validate shape ───────────────────────────────────────────────────
        if neural_data.ndim == 1:
            return jsonify({"error": f"Expected 2D array [T, 512], got shape {neural_data.shape}", "success": False}), 400

        neural_data = neural_data.astype(np.float32)

        # ── Inference ────────────────────────────────────────────────────────
        model, meta = get_model()
        inf = run_inference(model, meta, neural_data)
        phonemes = inf["phonemes"]

        # ── LLM translation ──────────────────────────────────────────────────
        text = "[no phonemes decoded]"
        if phonemes.strip():
            try:
                result = phonemes_to_text(phonemes)
                text = result["text"]
            except Exception as llm_err:
                text = f"[LLM error: {llm_err}]"

        return jsonify({
            "phonemes": phonemes,
            "text": text,
            "inference_time_ms": round(inf["inference_ms"], 1),
            "sequence_length": inf["seq_len"],
            "per": round(meta["best_per"] * 100, 1) if meta["best_per"] else 0.0,
            "success": True,
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e), "success": False}), 503
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/decode", methods=["POST"])
def decode():
    """
    Alternative endpoint: JSON body with neural_data 2-D array.
    Delegates to /api/predict logic.
    """
    return predict()


@app.route("/api/phonemes-to-text", methods=["POST"])
def convert_phonemes():
    """
    Convert an ARPAbet phoneme string to English text directly.

    Body: { "phonemes": "DH AH | K AE T ..." }
    """
    try:
        data = request.get_json()
        if not data or "phonemes" not in data:
            return jsonify({"error": "Missing 'phonemes' field", "success": False}), 400

        result = phonemes_to_text(data["phonemes"])
        return jsonify({
            "text": result["text"],
            "raw_phonemes": result["raw_phonemes"],
            "success": True,
        })
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/demo", methods=["GET"])
def demo():
    """
    Demo endpoint: convert pre-baked phoneme samples via the LLM layer.
    """
    samples = [
        "DH AH | K AE T | S AE T | AA N | DH AH | M AE T",
        "HH AW | AA R | Y UW | T AH D EY",
        "AY | W AA N T | T UW | G OW | HH OW M",
        "DH IH S | IH Z | AH | T EH S T | AH V | DH AH | B R EY N | T UW | T EH K S T | S IH S T AH M",
    ]
    try:
        results = batch_phonemes_to_text(samples)
        return jsonify({"results": results, "success": True})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


# ─── Startup banner ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  Brain2Text API Server")
    print("=" * 55)

    has_key = bool(os.environ.get("POLLINATION_API_KEY"))
    print(f"  Pollination API Key : {'✅ Configured' if has_key else '❌ Missing (.env)'}")

    pkl = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "model_GRU_pretrained.pkl")
    has_pkl = os.path.exists(pkl)
    print(f"  GRU Model (.pkl)    : {'✅ Found' if has_pkl else '❌ Not found'}")

    print(f"\n  Endpoints:")
    print(f"    GET  /api/health          - Server status")
    print(f"    POST /api/predict         - .npy or JSON → phonemes + text")
    print(f"    POST /api/decode          - Alias for /api/predict")
    print(f"    POST /api/phonemes-to-text- Phonemes → English text")
    print(f"    GET  /api/demo            - Demo phoneme conversion")
    print("=" * 55 + "\n")

    if has_pkl:
        print("Pre-loading model...")
        get_model()

    app.run(debug=True, port=5050)
