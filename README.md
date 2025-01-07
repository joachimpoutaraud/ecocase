# XCNet

## Requirements

```
conda create -n darksound python=3.10
conda install pip
pip install -r requirements.txt
```

Note: you need to have `ffmpeg` installed, because `librosa` relies on `ffmpeg` to handle MP3 files since it doesn’t support MP3 natively.
You can install `ffmpeg` on your computer by following these [instructions](https://github.com/fourMs/MGT-python/wiki/0-%E2%80%90-Installation#os-specific-installation-instructions)

## Workflow

```python
from xcnet import XCNet

# Create a Xeno-Canto network
xcnet = XCNet(save_path='train_data') # create 'train_data' folder to save data
# Build Xeno-Canto dataset with Xeno-Canto parameters from the config file (config.py)
xcnet.dataset(csv_filename="xc_dataset.csv") 
# Train model on top of BirdNET with Xeno-Canto dataset and save to disk
xcnet.train(model_output_path="checkpoints/custom/XCNet.tflite", csv_filename='train_dataset.csv')

# Reload Xeno-Canto network with custom classifier
xcnet = XCNet(save_path='<path_to_your_test_folder>', custom_classifier_path='checkpoints/custom/XCNet.tflite')
# Extract embeddings from audio files in 'test_data' folder with custom XCNet
xcnet.embeddings(dataset='<path_to_your_test_folder>', csv_filename='embeddings.csv') 
# Pseudo-label data and display clusters
xcnet.pseudo_label(display=True, csv_filename='cluster.csv') 
```