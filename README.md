# XCNet

## Requirements

```
conda create --name xcnet python=3.11 && conda activate xcnet && pip install -r requirements.txt
```

Note: you need to have `ffmpeg` installed, because `librosa` relies on `ffmpeg` to handle MP3 files since it doesn’t support MP3 natively.
You can install `ffmpeg` on your computer by following these [instructions](https://github.com/fourMs/MGT-python/wiki/0-%E2%80%90-Installation#os-specific-installation-instructions)

## Workflow

```python
from xcnet import XCNet

# Create instance of XenoCanto class
xcnet = XCNet()

# Query Xeno-Canto database and build dataset from metadata
xcnet.data(csv_filename='xc_dataset.csv')
# Label training data using clustering and save to cache
xcnet.labeling(method='clustering', save_cache="train_cache.npz")
# Download audio files from the training dataset
xcnet.download()
```