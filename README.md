# BirdMAP

## Requirements

```
conda create --name birdmap python=3.11 && conda activate birdmap && pip install -r requirements.txt
```

Note: you need to have `ffmpeg` installed, because `librosa` relies on `ffmpeg` to handle MP3 files since it doesn’t support MP3 natively.
You can install `ffmpeg` on your computer by following these [instructions](https://github.com/fourMs/MGT-python/wiki/0-%E2%80%90-Installation#os-specific-installation-instructions)

## Workflow

```python
from birdmap import BirdMAP

# Create a bird mapper
mapper = BirdMAP(save_path='train_data') # create 'train_data' folder to save data
# Build Xeno-Canto dataset with Xeno-Canto parameters from the config file (config.py)
mapper.dataset(csv_filename="xc_dataset.csv") 
   
# Extract embeddings from BirdNET model
mapper.embeddings(csv_filename='embeddings.csv') 
# Find cluster and assign pseudo-labels
mapper.pseudo_label(csv_filename='pseudo_label.csv')
# Train model with pseudo-labeled data and save model
mapper.train(model_output_path='checkpoints/custom/XCNet.tflite', csv_filename='train_dataset.csv', save_model=True)


# Reload Xeno-Canto network with custom classifier
mapper = BirdMAP(save_path='<path_to_your_test_folder>', custom_classifier_path='checkpoints/custom/XCNet.tflite')
# Extract embeddings from audio files in 'test_data' folder with custom XCNet
mapper.embeddings(dataset='<path_to_your_test_folder>', csv_filename='embeddings.csv') 
# Pseudo-label data and display clusters
mapper.pseudo_label(display=True, csv_filename='cluster.csv') 
```