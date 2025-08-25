import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
from IPython import get_ipython

import umap
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import factor_cmap
from bokeh.palettes import Category20

import warnings
warnings.filterwarnings(action="ignore")
import config as cfg


# def create_thumbnails(row, save_path, encode=False):
#     import librosa
#     import librosa.display
#     import matplotlib.pyplot as plt
#     from io import BytesIO
#     import base64
#     import soundfile as sf
#     import os

#     # TODO update spectrogram with FFmpeg and add option only audio or image
#     y, sr = load_audio(path=row.file, sr=16000, offset=row.start, duration=row.end-row.start, fmin=row.fmin, fmax=row.fmax, denoise=False, noise_profile=None)

#     S = librosa.feature.melspectrogram(y=y, sr=sr)
#     S_db = librosa.power_to_db(S, ref=np.max)
#     plt.figure(figsize=(4, 2))
#     librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
#     plt.axis('off')
        
#     if encode:
#         # Encode image
#         img_buf = BytesIO()
#         plt.savefig(img_buf, format='png')
#         img_buf.seek(0)
#         img_str = base64.b64encode(img_buf.read()).decode('utf-8')
#         plt.close()

#         # Encode audio
#         audio_buf = BytesIO()
#         sf.write(audio_buf, y, sr, format='wav')
#         audio_buf.seek(0)
#         audio_str = base64.b64encode(audio_buf.read()).decode('utf-8')

#         return f'data:image/png;base64,{img_str}', f'data:audio/wav;base64,{audio_str}'
    
#     else:           
#         if save_path is None:
#             save_path = os.getcwd()
        
#         save_path = os.path.join(save_path, 'thumbnails')
#         # Check if the folder exists, and create it if it doesn't
#         if not os.path.exists(save_path):
#             os.makedirs(save_path) 

#         # Check if row.file is a URL or a local file
#         if str(row.file).startswith(('http://', 'https://')):
#             filename = os.path.splitext(os.path.basename(os.path.dirname(row.file)))[0]
#         else:
#             filename = os.path.splitext(os.path.basename(row.file))[0]

#         # Save image to disk
#         img_filename = f"{str(filename)}_{str(row.start)}_{str(row.end)}_{str(row.fmin)}_{str(row.fmax)}.png"
#         plt.savefig(os.path.join(save_path, img_filename), format='png', bbox_inches='tight', pad_inches=0)
#         plt.close()

#         # Save audio to disk
#         audio_filename = f"{str(filename)}_{str(row.start)}_{str(row.end)}_{str(row.fmin)}_{str(row.fmax)}.wav"
#         sf.write(os.path.join(save_path, audio_filename), y, sr, format='wav')

#         # Only return file paths from thumbnails folder for writing html file
#         return os.path.join('thumbnails', img_filename), os.path.join('thumbnails', audio_filename)


def interactive_display(embeddings, labels, thumbnails=True, encode=False, frame_width=1000, frame_height=600, point_size=5, html_filename='scatterplot.html'):  

    # Get the embeddings    
    X = np.array([item['embeddings'] for item in embeddings])
    # Remove potential NaN values from the dataset
    X = np.nan_to_num(X)

    if X.shape[1] > 2:
        # Fit the UMAP model and transform the data
        reducer = umap.UMAP(n_components=2, metric='euclidean', random_state=2025)
        labels_arr = np.asarray(labels).astype(str)
        mask = labels_arr != 'Unknown'
        y = np.full(labels_arr.shape, -1, dtype=int)
        if mask.any():
            y_masked = pd.factorize(labels_arr[mask])[0]
            y[mask] = y_masked

        X = reducer.fit_transform(X, y=y)

    # Get the unique cluster labels and try to get optimal color number
    unique_clusters = np.unique(labels).astype(str).tolist()
    x, y = X[:, 0], X[:, 1]

    try:
        colors = Category20[len(unique_clusters)]
    except:
        colors = Category20[20]
        
    mapper = factor_cmap(field_name='label', palette=colors, factors=unique_clusters)

    if 'ZMQInteractiveShell' in get_ipython().__class__.__name__:
        from bokeh.io import output_notebook
        output_notebook() 
    
    # if thumbnails:
    #     from concurrent.futures import ProcessPoolExecutor
    #     from tqdm import tqdm
    #     import os
        
    #     spectrograms, audio_files = [], []

    #     with tqdm(total=len(df), desc="Process image and audio thumbnails") as pbar: 
    #         with ProcessPoolExecutor(max_workers=cfg.CPU_THREADS) as executor:
    #             futures = [executor.submit(create_thumbnails, row, save_path, encode) for _, row in df.iterrows()]
            
    #             for future in futures:
    #                 pbar.update(1)
    #                 try:
    #                     img_str, audio_file = future.result()
    #                     spectrograms.append(img_str)
    #                     audio_files.append(audio_file)
    #                 except:
    #                     pass

    #     data_dict = dict(x=x, 
    #              y=y, 
    #              label=labels.astype(str).tolist(), 
    #              spectrogram=spectrograms, 
    #              audio=audio_files, 
    #              filename=audio_files)

    #     tooltips = """
    #         <div>
    #         <div>
    #             <img src="@spectrogram" height="200" width="200" border="1" style="float: left; margin: 0px 15px 15px 0px;"></img>
    #         </div>
    #         <div style="font-size: 12px; color: #966;">Filename: @filename</div>
    #         <div style="font-size: 12px; color: #966;">Label: @label</div>
    #     """

    #     if set(['lat', 'lng']).issubset(df.columns):
    #         data_dict.update(lat=df['lat'].astype(str).tolist(), 
    #                  lng=df['lng'].astype(str).tolist())
    #         tooltips += """
    #         <div style="font-size: 12px; color: #966;">Latitude: @lat</div>
    #         <div style="font-size: 12px; color: #966;">Longitude: @lng</div>
    #         """
            
    #     if 'confidence' in df.columns:
    #         data_dict.update(confidence=df['confidence'].to_numpy())
    #         tooltips += """
    #         <div style="font-size: 12px; color: #966;">Confidence: @confidence</div>
    #         """

    #     tooltips += """
    #         </div>
    #         <div>
    #             <audio autoplay controls style="width: 200px;">
    #             <source src="@audio" type="audio/wav">
    #             Your browser does not support the audio element.
    #             </audio>
    #         </div>
    #         </div>
    #     """

    #     source = ColumnDataSource(data=data_dict)
    #     hover = HoverTool(tooltips=tooltips)

    #     p = figure(frame_width=frame_width, frame_height=frame_height, title=f"{cfg.METHOD} interactive scatter plot with spectrograms and audio", tools="pan,wheel_zoom,reset")
    #     if 'confidence' in df.columns and 'confidence' in source.data:
    #         p.scatter('x', 'y', size=point_size, color=mapper, fill_alpha='confidence', source=source)
    #     else:
    #         p.scatter('x', 'y', size=point_size, color=mapper, source=source)
    #     p.add_tools(hover)
        
    # else:
    labels_arr = np.array(labels).astype(str)
    unique_labels = np.unique(labels_arr).tolist()
    # Ensure we have enough colors (repeat if necessary)
    palette = colors if len(colors) >= len(unique_labels) else (colors * ((len(unique_labels) // len(colors)) + 1))

    p = figure(frame_width=frame_width, frame_height=frame_height, title=f"{cfg.METHOD} interactive scatter plot", tools="pan,wheel_zoom,reset")

    for i, lbl in enumerate(unique_labels):
        mask = labels_arr == lbl
        if not np.any(mask):
            continue
        src = ColumnDataSource(data=dict(x=x[mask], y=y[mask], label=[lbl] * mask.sum()))
        p.scatter('x', 'y', size=point_size, color=palette[i], source=src, legend_label=str(lbl))

    p.legend.location = "top_right"
    p.legend.click_policy = "hide"

    output_file(html_filename)
    show(p)