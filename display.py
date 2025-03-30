import numpy as np
import pandas as pd
import umap
from IPython import get_ipython

from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import factor_cmap
from bokeh.palettes import Category20

import config as cfg
from cluster import prepare_embeddings

import warnings
warnings.filterwarnings(action="ignore")

def create_thumbnails(row, save_path, encode=False):
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    import soundfile as sf
    import audio
    import os
    
    y, sr = audio.openAudioFile(row.file, sample_rate=cfg.BANDPASS_FMAX, offset=row.start, duration=row.end - row.start)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(4, 2))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
        
    if encode:
        # Encode image
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_str = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close()

        # Encode audio
        audio_buf = BytesIO()
        sf.write(audio_buf, y, sr, format='wav')
        audio_buf.seek(0)
        audio_str = base64.b64encode(audio_buf.read()).decode('utf-8')

        return f'data:image/png;base64,{img_str}', f'data:audio/wav;base64,{audio_str}'
    
    else:           
        # Ensure save_path is not None
        if save_path is None:
            raise ValueError("save_path cannot be None when encode is False")
        
        save_path = os.path.join(save_path, 'thumbnails')
        # Check if the folder exists, and create it if it doesn't
        if not os.path.exists(save_path):
            os.makedirs(save_path) 

        # Save image to disk
        img_filename = f"{os.path.join(save_path, str(row.categories) + str(row.start))}.png"
        plt.savefig(img_filename, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save audio to disk
        audio_filename = f"{os.path.join(save_path, str(row.categories) + str(row.start))}.wav"
        sf.write(audio_filename, y, sr, format='wav')

        return f"{os.path.join(save_path, str(row.categories) + str(row.start))}.png", f"{os.path.join(save_path, str(row.categories) + str(row.start))}.wav"


def interactive_display(df_cluster, output_filename='interactive_scatter', save_path=None, thumbnails=True, encode=False, noise=False, frame_width=1000, frame_height=600, point_size=10):  
    
    # Check if dataset is a path to a dataframe, audio file or folder
    if type(df_cluster) == str:
        import os
        ext = os.path.splitext(df_cluster)[1]
        # Check file extension
        if ext == ".csv": 
            df_cluster = pd.read_csv(df_cluster, sep=";")
    
    if not noise:  
        # Remove noise points from display
        df_cluster = df_cluster[df_cluster['pseudo_label'] > 0]
    
    # Prepare the embeddings    
    X = df_cluster.loc[:, df_cluster.columns.str.startswith("emb")].to_numpy()
    # Get all the cluster labels and filenames
    labels = df_cluster['cluster_number']
    
    # Fit the UMAP model
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=cfg.RANDOM_SEED, densmap=True)
    embedding = reducer.fit_transform(X)
    x, y = embedding[:, 0], embedding[:, 1]

    # Get the unique cluster labels and try to get optimal color number
    unique_clusters = labels.unique().astype(str).tolist()
    
    try:
        colors = Category20[len(unique_clusters)]
    except:
        colors = Category20[20]
        
    mapper = factor_cmap(field_name='label', palette=colors, factors=unique_clusters)

    if 'ZMQInteractiveShell' in get_ipython().__class__.__name__:
        from bokeh.io import output_notebook
        output_notebook() 
    
    if thumbnails:
        from concurrent.futures import ProcessPoolExecutor
        from tqdm import tqdm
        import os
        
        nb_cpu = os.cpu_count() 
        spectrograms, audio_files = [], []

        with tqdm(total=len(df_cluster), desc="Process image and audio thumbnails") as pbar: 
            with ProcessPoolExecutor(max_workers=nb_cpu-1) as executor:
                futures = [executor.submit(create_thumbnails, row, save_path, encode) for _, row in df_cluster.iterrows()]
            
                for future in futures:
                    pbar.update(1)
                    try:
                        img_str, audio_file = future.result()
                        spectrograms.append(img_str)
                        audio_files.append(audio_file)
                    except:
                        pass

        if set(['lat', 'lng']).issubset(df_cluster.columns):
            source = ColumnDataSource(data=dict(x=x, 
                                                y=y, 
                                                label=labels.astype(str).tolist(), 
                                                spectrogram=spectrograms, 
                                                audio=audio_files, 
                                                filename=df_cluster['file'].astype(str).tolist(), 
                                                lat=df_cluster['lat'].astype(str).tolist(), 
                                                lng=df_cluster['lng'].astype(str).tolist())
                                    )

            hover = HoverTool(tooltips="""
                <div>
                    <div>
                        <img src="@spectrogram" height="200" width="200" border="1" style="float: left; margin: 0px 15px 15px 0px;"></img>
                    </div>
                    <div style="font-size: 15px; font-weight: bold;">Label: @label</div>
                    <div style="font-size: 12px; color: #966;">Filename: @filename</div>
                    <div style="font-size: 12px; color: #966;">Latitude: @lat</div>
                    <div style="font-size: 12px; color: #966;">Longitude: @lng</div>
                    </div>
                    <div>
                        <audio autoplay controls style="width: 200px;">
                            <source src="@audio" type="audio/wav">
                            Your browser does not support the audio element.
                        </audio>
                    </div>
                </div>
                """
            )

        else:
            source = ColumnDataSource(data=dict(x=x, 
                                                y=y, 
                                                label=labels.astype(str).tolist(), 
                                                spectrogram=spectrograms, 
                                                audio=audio_files, 
                                                filename=df_cluster['file'].astype(str).tolist())
                                    )

            hover = HoverTool(tooltips="""
                <div>
                    <div>
                        <img src="@spectrogram" height="200" width="200" border="1" style="float: left; margin: 0px 15px 15px 0px;"></img>
                    </div>
                    <div style="font-size: 15px; font-weight: bold;">Label: @label</div>
                    <div style="font-size: 12px; color: #966;">Filename: @filename</div>
                    </div>
                    <div>
                        <audio autoplay controls style="width: 200px;">
                            <source src="@audio" type="audio/wav">
                            Your browser does not support the audio element.
                        </audio>
                    </div>
                </div>
                """
            )    

        p = figure(frame_width=frame_width, frame_height=frame_height, title=f"{cfg.METHOD} interactive scatter plot with spectrograms and audio", tools="pan,wheel_zoom,reset")
        p.scatter('x', 'y', size=point_size, color=mapper, source=source)
        p.add_tools(hover)
        
    else:
        source = ColumnDataSource(data=dict(x=x, y=y, label=labels.astype(str).tolist()))
        p = figure(frame_width=frame_width, frame_height=frame_height, title=f"{cfg.METHOD} interactive scatter plot", tools="pan,wheel_zoom,reset")
        p.scatter('x', 'y', size=point_size, color=mapper, source=source)
        
    output_file(f"{output_filename}.html")
    show(p)


if __name__ == "__main__":
    
    df_cluster = pd.read_csv("cluster.csv", sep=";")
    # Get interactive display of the clusters
    interactive_display(df_cluster, thumbnails=True)