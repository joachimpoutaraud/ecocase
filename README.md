# XCNet
 
```python

from xcnet import XCNet
# Create a Xeno-Canto network
xcnet = XCNet()

# Build Xeno-Canto dataset with Xeno-Canto parameters from the config file (config.py)
xcnet.dataset() 
# Fine-tune BirdNET with Xeno-Canto dataset and save custom classifier to disk
xcnet.fine_tuning(model_output_path="checkpoints/custom/XCNet.tflite", csv_filename='train_data.csv')
# Extract embeddings from audio files in 'test_data' folder with XCNet
xcnet.embeddings(dataset='test_data', csv_filename='embeddings.csv')    
# Pseudo-label data and display clusters
xcnet.pseudo_labeling(display=True, csv_filename='cluster.csv') 
```