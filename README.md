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
rec = RecordingROI(path='audio.wav', rate=48000, fmin=1, fmax=15000, multiprocessing=True)
# Detect region of interests (ROIs) in the recording
rois = rec.detect_rois(min_duration=0.5, max_duration=10.0, display=False)
# Segment ROIs into 3-second audio chunks
data = rec.segment_rois(rois, noise_reduction=-20, save_path='data.npy')


### Create RecordingAnalysis instance
rec = RecordingAnalysis(data, multiprocessing=True)
# Extract embeddings and predict species with BirdNET
embeddings = rec.birdnet_analysis(extract=True, predict=True, min_confidence=0.25, save_path='embeddings.npy')
# Analyze frequency range of the species
results = rec.frequency_analysis() # Chloris chloris_European Greenfinch, Turdus iliacus_Redwing
# # Analyze repertoire of the species
# repertoire = rec.repertoire_analysis(species='Chloris chloris_European Greenfinch') 

# Display rois in interactive plot
rec.display_rois(noise_reduction=-20)
```