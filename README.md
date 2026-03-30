# Sleep Assist – Smart Sleep Phase Alarm

A machine learning system that identifies optimal wake-up times by detecting favorable sleep phases from wearable sensor data.

## Overview

Sleep Assist combines transfer learning with personalized LSTM models to predict the best moment to wake up — minimizing sleep inertia and improving how you feel in the morning.

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 79.6% |
| F1-Score (weighted) | 79.1% |
| Precision (binary) | up to 90.6% |
| Recall (binary) | up to 93.7% |

**Smart Alarm example:** Optimal wake time found at 4:55 AM (91.1% confidence) vs. traditional alarm at 4:45 AM (77.7% confidence).

## How It Works

The pipeline runs in four phases:

1. **General model** — Cross-dataset training for a robust baseline
2. **Transfer learning** — Pseudo-labeling on personalized data (98% confidence threshold)
3. **Fine-tuning** — Frozen layers + new output head for individual sleep patterns
4. **Smart Alarm** — Real-time simulation to find the optimal wake window

## Dataset

Uses the **DREAMT dataset** with Empatica E4 wearable sensor data:
- Signals: BVP, ACC (X/Y/Z), EDA, TEMP, HR
- Task: Binary classification — favorable vs. unfavorable wake-up moments

## Requirements

- Python 3.8+
- TensorFlow 2.x
- scikit-learn, pandas, numpy, matplotlib
- 8 GB+ RAM (GPU recommended)

## Usage

**Recommended: Google Colab**

1. Upload the subject files `S002`, `S003`, ..., `S007` and the script to Colab
2. Run the notebook — training and simulation start automatically

**Local setup:**

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
python sleep_assist_final.py
```

CSV files must follow this format:
```
TIMESTAMP, BVP, ACC_X, ACC_Y, ACC_Z, TEMP, EDA, HR, Sleep_Stage
```

Training takes approximately 20–30 min on CPU or 5–10 min on GPU.

## Who Is This For?

People with irregular sleep schedules, shift workers, or anyone looking to improve their sleep quality and morning performance.

## Author

**Sanyukt Mishra** — sanyukt.mishra@outlook.com — 2024
