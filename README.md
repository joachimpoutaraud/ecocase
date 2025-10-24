# Ecological Clustering with Analysis of Sound Events (ECOCASE)

## Requirements

```
conda env create -f environment.yml
```

Note: EcoCASE requires `FFmpeg` to read and process audio files. Install `FFmpeg` and make sure it is available on your `PATH`.

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
contours = case.detect_rois(min_duration=0.2, max_duration=10.0, display=False)
# Segment region of interests (ROIs) in the recording and save to binary file
rois = case.segment_rois(contours, noise_reduction=-20, binary_file_path='rois.npy')

# Extract embeddings and predict species with BirdNET and save to binary file
embeddings = case.birdnet_analysis(embeddings=True, predict=True, min_confidence=0.80, binary_file_path='embeddings.npy')
# Analyze frequency range of the predicted species
results = case.frequency_analysis()
# If there is enough data on one of the predicted species, analyze the repertoire of the selected species
# repertoire = case.repertoire_analysis(species='Chloris chloris_European Greenfinch')

# Display clusters in an interactive plot
case.display_clusters(noise_reduction=-20)
```