# Data Pipeline

## 1. Groove MIDI Dataset

Download:
https://magenta.withgoogle.com/datasets/groove

Place into:
data/raw/groove/

Run:
python3 data/process_groove.py

Output:
data/processed/groove_grids.npy

Shape:
(N, 4, 16)


## 2. Freesound Sample Library

Get API key:
https://freesound.org/apiv2/apply/

Set environment variable:
export FREESOUND_API_KEY=your_key

Run:
python3 data/download_samples.py

Output:
data/samples/
data/samples/manifest.json


## 3. Directory Structure

data/
  raw/groove/
  samples/
  processed/