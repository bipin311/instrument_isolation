# Musical Instrument (Source) Separation with U-Net (MUSDB18)

A lightweight PyTorch project to isolate musical sources (e.g. vocals, bass, drums) from polyphonic mixtures using a 2D U-Net operating on magnitude spectrograms. Includes:
- Data preprocessing (STFT magnitude extraction from MUSDB18 stems)
- U-Net training scripts per target
- Inference to reconstruct estimated sources
- Simple SDR evaluation and comparison with a Demucs baseline

---

## 1. Project Goals
- Learn a mask (or direct magnitude) that recovers an individual instrument from a mixture.
- Compare a simple U-Net vs a state-of-the-art baseline (Demucs).
- Provide clear, minimal, reproducible code.

---

## 2. Dataset: MUSDB18
- 150 songs (100 train / 50 test) with stems: mixture, vocals, drums, bass, other.
- Sample rate: 44.1 kHz. Provided in `.stem.mp4`.
- Not included in this repo — you must download separately (see: https://sigsep.github.io/datasets/musdb.html).
