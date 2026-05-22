"""
inference.py — Brain2Text GRU model loader + inference pipeline.

The .pkl file (model_GRU_pretrained.pkl) is a Python dict with the schema:
    {
        'experiment_name': str,
        'best_per': float,
        'best_val_loss': float,
        'history': dict,
        'model_state_dict': OrderedDict,   # keys: adapter.*, rnn.*, fc.*
        'arch': {
            'model_type': 'GRU',
            'data_input_size': 512,
            'adapter_output_size': 256,
            'hidden_size': 768,
            'output_size': 41,
            'num_layers': 5,
            'bidirectional': False,
            'vocab': [...],
            'blank_id': 0
        }
    }
"""

import pickle
import os
import time

import numpy as np
import torch
import torch.nn as nn

# ─── Default model path ──────────────────────────────────────────────────────

PKL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "model_GRU_pretrained.pkl",
)


# ─── Model architecture (mirrors the Kaggle notebook exactly) ────────────────

class GRUBrainDecoder(nn.Module):
    """
    Adapter → multi-layer GRU → FC → log_softmax.

    State-dict key names: adapter.*, rnn.*, fc.*
    """

    def __init__(self, data_input_size, adapter_output_size,
                 hidden_size, output_size, num_layers, bidirectional):
        super().__init__()
        self.adapter = nn.Linear(data_input_size, adapter_output_size)
        self.rnn = nn.GRU(
            input_size=adapter_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        fc_in = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_in, output_size)

    def forward(self, x):
        x = self.adapter(x)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return nn.functional.log_softmax(out, dim=2)


# ─── CTC greedy decoder ──────────────────────────────────────────────────────

def greedy_decode(logits, vocab, blank_id=0):
    """
    CTC greedy decode: argmax → collapse consecutive → remove blanks.

    Args:
        logits: numpy array [T, num_classes]  (log-softmax or raw)
        vocab:  list of phoneme strings (index 1-based; 0 = blank)
        blank_id: int (default 0)

    Returns:
        str — space-separated phonemes with '|' word boundaries
    """
    # Build token map: {1: 'AA', 2: 'AE', ...}
    token_map = {i + 1: p for i, p in enumerate(vocab)}
    token_map[blank_id] = ""

    pred = np.argmax(logits, axis=-1)

    # Collapse consecutive duplicates
    collapsed = []
    prev = None
    for idx in pred:
        if idx != prev:
            collapsed.append(int(idx))
        prev = idx

    # Remove blanks and map to phonemes
    phonemes = [token_map[i] for i in collapsed if i != blank_id and i in token_map]
    return " ".join(phonemes)


# ─── Public API ──────────────────────────────────────────────────────────────

def load_model(model_path=None):
    """
    Load the trained GRU model from the .pkl file.

    Returns a tuple: (model: nn.Module, meta: dict)
    meta contains: 'vocab', 'blank_id', 'arch', 'best_per'
    """
    path = model_path or PKL_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Make sure 'model_GRU_pretrained.pkl' is in the project root."
        )

    with open(path, "rb") as f:
        ckpt = pickle.load(f)

    arch = ckpt["arch"]

    model = GRUBrainDecoder(
        data_input_size=arch["data_input_size"],
        adapter_output_size=arch["adapter_output_size"],
        hidden_size=arch["hidden_size"],
        output_size=arch["output_size"],
        num_layers=arch["num_layers"],
        bidirectional=arch["bidirectional"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = {
        "vocab": arch["vocab"],
        "blank_id": arch.get("blank_id", 0),
        "arch": arch,
        "best_per": ckpt.get("best_per", None),
    }

    print(f"✅ Model loaded: {arch['model_type']} | "
          f"hidden={arch['hidden_size']} | layers={arch['num_layers']} | "
          f"best_per={meta['best_per']:.4f}")
    return model, meta


def run_inference(model, meta, neural_data):
    """
    Run inference on neural data.

    Args:
        model:        nn.Module (from load_model)
        meta:         dict (from load_model)
        neural_data:  np.ndarray, shape [T, 512]

    Returns:
        dict with keys: phonemes (str), inference_ms (float), seq_len (int)
    """
    if neural_data.ndim == 2:
        x = torch.tensor(neural_data, dtype=torch.float32).unsqueeze(0)  # [1, T, 512]
    elif neural_data.ndim == 3:
        x = torch.tensor(neural_data, dtype=torch.float32)
    else:
        raise ValueError(f"Expected 2D or 3D neural_data, got shape {neural_data.shape}")

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)  # [1, T, 41]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logits_np = logits[0].numpy()  # [T, 41]
    phonemes = greedy_decode(logits_np, meta["vocab"], meta["blank_id"])

    return {
        "phonemes": phonemes,
        "inference_ms": elapsed_ms,
        "seq_len": int(x.shape[1]),
    }


# ─── CLI smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading model...")
    model, meta = load_model()

    # Synthetic input: T=100 timesteps, 512 features
    dummy = np.random.randn(100, 512).astype(np.float32)
    result = run_inference(model, meta, dummy)

    print(f"Phonemes : {result['phonemes']}")
    print(f"Inf time : {result['inference_ms']:.1f} ms")
    print(f"Seq len  : {result['seq_len']}")
