# XCNet
 
```python

from xcnet import XCNet
# Create a Xeno-Canto network
xcnet = XCNet(save_path='train_data')   
# Build Xeno-Canto dataset with Xeno-Canto parameters from the config file (config.py)
xcnet.dataset(csv_filename="xc_data.csv") 
# Train model on top of BirdNET with Xeno-Canto dataset and save custom classifier to disk
xcnet.train(model_output_path="train_data/XCNet.tflite", csv_filename='pseudo_data.csv')

# # Reload Xeno-Canto network with custom classifier
# xcnet = XCNet(custom_classifier_path="train_data/XCNet.tflite")
# Extract embeddings from audio files in test data folder with custom XCNet
xcnet.embeddings(dataset='<path to your test data folder>', csv_filename='embeddings.csv')    
# Pseudo-label data and display clusters
xcnet.pseudo_label(display=True, csv_filename='cluster.csv') 
```