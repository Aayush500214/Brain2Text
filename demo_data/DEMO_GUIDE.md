# Brain2Text — Demo Dataset Cheat Sheet

> **Important:** The GRU model was trained on real EEG brain signals (t15 copyTask dataset).  
> When fed **random noise**, it always predicts `AY |` ("I") — this is expected behaviour at 95% PER.  
> For a live demo, use **DEMO MODE** in the frontend (pre-baked phonemes → decoder → text),  
> or upload real `.hdf5` session data converted to `.npy`.

---

## How to use the demo `.npy` files

These files are in `demo_data/`. Each is a `[T, 512]` float32 array of **synthetic** neural data.  
Upload them in **LIVE API MODE** (`http://localhost:5050`) — the model will run inference on them.

Because the model outputs `AY |` for noise, the decoded text will be **"I."**  
This is the honest result of the model on out-of-distribution data.

---

## What to show during the presentation

### Option A — Demo Mode (recommended)
Use the built-in **DEMO MODE** dropdown in the frontend.  
The phonemes are pre-baked from the competition dataset phoneme vocabulary.

| Trial | Pre-baked phonemes | Expected text |
|-------|--------------------|---------------|
| Trial 1 | `W AO T ER \| B AA T AH L` | **Water bottle.** |
| Trial 2 | `HH AH L OW \| W ER L D` | **Hello world.** |
| Trial 3 | `G UH D \| M AO R N IH NG` | **Good morning.** |
| Trial 4 | `OW P AH N \| DH AH \| D AO R` | **Open the door.** |
| Trial 5 | `TH AE NG K \| Y UW` | **Thank you.** |
| Trial 6 | `N UH R AH L \| D IH K OW D` | **Neural decode.** |

All of these now decode **correctly** using the CMU Pronouncing Dictionary (no LLM hallucination).

---

### Option B — Live API Mode with `/api/phonemes-to-text`
Test the decoder directly with:

```bash
curl -X POST http://localhost:5050/api/phonemes-to-text \
  -H "Content-Type: application/json" \
  -d '{"phonemes": "OW P AH N | DH AH | D AO R"}'
# → { "text": "Open the door." }
```

---

### Option C — Upload a `.npy` file
Shape must be `[T, 512]` float32. Real EEG recordings from the t15 copyTask dataset work best.

```python
import numpy as np

# Example: save a real HDF5 trial as .npy
import h5py
with h5py.File("data_test.hdf5", "r") as f:
    trial = list(f.keys())[0]
    features = f[trial]["input_features"][:]   # shape [T, 512]
    np.save("my_trial.npy", features.astype("float32"))
```

Then upload `my_trial.npy` in the LIVE API MODE tab.

---

## Pipeline Summary

```
.npy [T, 512]
    ↓  Adapter  Linear(512 → 256)
    ↓  GRU      (256 → 768 hidden, 5 layers)
    ↓  FC       (768 → 41 classes)
    ↓  CTC Greedy Decode
Phonemes: "W AO T ER | B AA T AH L"
    ↓  CMU Pronouncing Dictionary lookup
Text: "Water bottle."
```

| Component | Detail |
|-----------|--------|
| Model | GRU (5-layer, 768 hidden) |
| Input | 512-dim neural features per 20ms bin |
| Output vocab | 40 ARPAbet phonemes + blank |
| Decoder | CTC greedy + CMU dict (word-frequency ranked) |
| Best PER | 95.03% (training, Kaggle Brain-to-Text '25) |
| Inference time | ~40–70 ms on CPU |
