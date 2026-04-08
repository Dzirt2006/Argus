# Wake Word R&D

Custom wake word training pipeline for Argus using OpenWakeWord.

## Overview

Train a custom wake word model (e.g., "hey argus") that works reliably across
multiple speakers and acoustic environments. The output is a small ONNX model
(~200 KB) that drops into the Argus voice pipeline.

## Structure

```
rnd/wakeword/
├── config.yaml                  # global config (wake phrase, audio format, paths)
├── requirements.txt             # Python dependencies
├── guide.md                     # recording guide (read this first!)
│
├── dataset/                     # Pipeline 1: dataset creation
│   ├── record_positive.py       # record wake word samples from mic
│   ├── record_negative.py       # record ambient, speech, adversarial phrases
│   ├── import_external.py       # import + normalize files from phones/laptops
│   ├── generate_synthetic.py    # bulk TTS generation via Piper (904 speakers)
│   ├── download_negatives.py    # download MUSAN noise, RIR datasets
│   ├── augment.py               # noise, reverb, speed, volume augmentation
│   └── validate_dataset.py      # check format, quality, balance, report stats
│
├── training/                    # Pipeline 2: model training
│   ├── config.yaml              # training hyperparameters
│   ├── extract_features.py      # extract OWW embeddings from audio
│   ├── train.py                 # train DNN/RNN classifier
│   ├── evaluate.py              # metrics, ROC curve, optimal threshold
│   └── export_onnx.py           # export to ONNX for deployment
│
├── data/                        # created at runtime (gitignored)
│   ├── positive/                # real mic recordings by speaker
│   ├── negative/                # ambient, conversation, adversarial
│   ├── synthetic/               # Piper TTS generated samples
│   ├── external/                # imported from other devices
│   ├── augmented/               # augmented copies
│   ├── noise/                   # background noise (MUSAN)
│   ├── rir/                     # room impulse responses
│   └── features/                # extracted embeddings (.npy)
│
└── models/                      # trained models (gitignored)
    ├── best_model.pt            # best PyTorch checkpoint
    ├── hey_argus.onnx           # exported ONNX for deployment
    ├── training_history.json    # loss/metrics per epoch
    └── evaluation_results.json  # test metrics at all thresholds
```

## Quick Start

### 1. Setup

```bash
cd /opt/projects/Argus
pip install -r rnd/wakeword/requirements.txt
```

### 2. Read the recording guide

Read `guide.md` before recording — it covers microphone tips, what variations
to include, adversarial phrases, and how to record from external devices.

### 3. Create the dataset

Run these scripts from the IDE (each has `if __name__ == "__main__"`):

```
# Step 1: Record real samples (20-50 per speaker)
python rnd/wakeword/dataset/record_positive.py --speaker alex --count 50

# Step 2: Record negative samples (ambient + adversarial phrases)
python rnd/wakeword/dataset/record_negative.py --mode all

# Step 3: Import recordings from phone/laptop (optional)
python rnd/wakeword/dataset/import_external.py

# Step 4: Generate synthetic samples (10,000+)
python rnd/wakeword/dataset/generate_synthetic.py --count 10000

# Step 5: Download noise/RIR datasets for augmentation
python rnd/wakeword/dataset/download_negatives.py

# Step 6: Augment all positive samples
python rnd/wakeword/dataset/augment.py

# Step 7: Validate everything
python rnd/wakeword/dataset/validate_dataset.py
```

### 4. Train the model

```
# Extract embeddings from audio
python rnd/wakeword/training/extract_features.py

# Train the classifier
python rnd/wakeword/training/train.py

# Evaluate (metrics, ROC curve, optimal threshold)
python rnd/wakeword/training/evaluate.py

# Export to ONNX
python rnd/wakeword/training/export_onnx.py
```

### 5. Deploy

Copy the ONNX model and update `.env`:

```
VOICE_WAKEWORD_MODEL=rnd/wakeword/models/hey_argus.onnx
VOICE_WAKEWORD_THRESHOLD=0.5   # adjust based on evaluate.py output
```

## Configuration

Edit `config.yaml` to change:
- **wake_phrase** — the phrase to detect (default: "hey argus")
- **audio format** — sample rate, duration limits
- **synthetic generation** — Piper model, speaker count, speed range
- **augmentation** — noise levels, RIR probability, speed/volume ranges

Edit `training/config.yaml` to change:
- **model architecture** — DNN vs RNN, hidden dim, layers, dropout
- **training** — epochs, batch size, learning rate, early stopping
- **evaluation** — threshold sweep, target FAR/FRR
