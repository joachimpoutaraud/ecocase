# XCNet

## Requirements

```
conda env create -f environment.yml
```

Note: you need to have `ffmpeg` installed.
You can install `ffmpeg` on your computer by following these [instructions](https://github.com/fourMs/MGT-python/wiki/0-%E2%80%90-Installation#os-specific-installation-instructions)

## Workflow

```python
from main import RecordingROI

### Create RecordingROI instance
rec = RecordingROI(path="audio.wav", rate=48000, fmin=150, fmax=15000, multiprocessing=True)

# Detect region of interests (ROIs) in the recording
rois = rec.detect_rois(min_duration=0.5, max_duration=10.0, display=False)
# Segment ROIs into 3-second audio chunks
segments = rec.segment_rois(rois, noise_reduction=-20, download=False)

# Extract embeddings and predict species with BirdNET
data = rec.birdnet_analysis(segments, extract=True, predict=True, min_confidence=0.75)
# Analyze frequency range for each detected species
results = rec.frequency_analysis(data)
# Display repertoire in interactive plot
rec.display_rois(data, noise_reduction=-50)
```