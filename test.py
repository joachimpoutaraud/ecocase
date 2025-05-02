import pandas as pd
import config as cfg
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import audio


species_num = 5

df_train = pd.read_csv('train_data/pseudo_label.csv', sep=";", low_memory=False)
df_train = df_train[df_train.pseudo_label == 1].reset_index(drop=True)

fig, axes = plt.subplots(6, 5, figsize=(20, 12)) 
length = len(df_train[df_train["categories"] == cfg.SPECIES_XC[species_num]])
fig.suptitle(f"{str(cfg.SPECIES_XC[species_num])}: {str(length)}")

for i, (_, row) in enumerate(df_train[df_train["categories"] == cfg.SPECIES_XC[species_num]].iloc[:30].iterrows()):
    y, sr = audio.openAudioFile(row.file, sample_rate=16000, offset=row.start, duration=row.end - row.start)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    ax = axes[i // 5, i % 5]  # Map index to 2D grid
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.axis('off')

fig.colorbar(img, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1)
plt.show()


# df_test = pd.read_csv('test_data/cluster2.csv', sep=";", low_memory=False)
# fig, axes = plt.subplots(6, 5, figsize=(20, 12))  # 2 rows, 5 columns

# for i, (_, row) in enumerate(df_test[df_test["cluster_number"] == 0].iloc[:30].iterrows()):
#     y, sr = audio.openAudioFile(row.file, sample_rate=16000, offset=row.start, duration=row.end - row.start)
#     S = librosa.feature.melspectrogram(y=y, sr=sr)
#     S_db = librosa.power_to_db(S, ref=np.max)
#     ax = axes[i // 5, i % 5]  # Map index to 2D grid
#     img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
#     ax.axis('off')

# fig.colorbar(img, ax=axes, orientation='horizontal', fraction=0.02, pad=0.1)
# plt.show()


