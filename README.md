# XCNet

## Requirements

```
conda env create -f environment.yml
```

Note: you need to have `ffmpeg` installed.
You can install `ffmpeg` on your computer by following these [instructions](https://github.com/fourMs/MGT-python/wiki/0-%E2%80%90-Installation#os-specific-installation-instructions)

## Workflow

```python
from main import RecordingROI, RecordingAnalysis

### Create RecordingROI instance
rec = RecordingROI(path='audio.wav', multiprocessing=True, save_path='rois.npy')
# Detect region of interests (ROIs) in the recording
rois = rec.detect_rois(min_duration=0.5, max_duration=10.0, display=False)
# Segment region of interests (ROIs) in the recording
data = rec.segment_rois(rois, noise_reduction=-20)


### Create RecordingAnalysis instance
rec = RecordingAnalysis(data=data, multiprocessing=True, save_path='analysis.npy')
# Extract embeddings and predict species with BirdNET
embeddings = rec.birdnet_analysis(extract=True, predict=True, min_confidence=0.75)
# Analyze frequency range of the predicted species
results = rec.frequency_analysis()

# Display rois in interactive plot
rec.display_rois(noise_reduction=-20)
```