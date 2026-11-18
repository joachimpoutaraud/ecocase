# Ecological Clustering with Analysis of Sound Events (EcoCASE)

## Requirements

```bash
conda env create -f environment.yml
conda activate ecocase
```

EcoCASE requires `FFmpeg` to read and process audio files. Install `FFmpeg` and make sure it is available on your `PATH`.

Verify the installation with:

```bash
ffmpeg -version
```

For platform-specific installation instructions, see: https://github.com/fourMs/MGT-python/wiki/0-%E2%80%90-Installation#os-specific-installation-instructions

## Workflow

```python
from main import EcoCASE

# Initialize EcoCASE with a file or folder path containing audio files
case = EcoCASE(path='<path_to_your_file>', multiprocessing=True, rate=48000, fmin=1, fmax=15000)

# Detect region of interests (ROIs) in the recording
contours = case.detect_rois(min_duration=0.5, max_duration=10.0, display=False)
# Segment region of interests (ROIs) in the recording
rois = case.segment_rois(contours, save_binary='rois.npy')

# Extract embeddings and predict species with BirdNET
data = case.birdnet_analysis(rois, embeddings=True, predict=True, save_binary='data.npy')
# Analyze frequency range of the predicted species
results = case.frequency_analysis()

# Display ecological clusters in an interactive plot
case.display_clusters(port=8080)
```
