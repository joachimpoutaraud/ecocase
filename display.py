import subprocess
import io, base64
import numpy as np
from scipy.io import wavfile
import webbrowser

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Button, Div, CustomJS, TapTool
from bokeh.layouts import column, row
from bokeh.palettes import Category20
from bokeh.server.server import Server

from rois import extract_roi_audio

import warnings
warnings.filterwarnings("ignore")

def reducer(data, min_confidence, random_state):
    
    # Filter out low-confidence predictions by marking them as "Unknown"
    if min_confidence is not None:
        for item in data:
            conf = item.get('confidence')
            if conf < float(min_confidence):
                if 'repertoire' in data[0]:
                    item['repertoire'] = 'Unknown'
                else:
                    item['prediction'] = 'Unknown'
                item['confidence'] = 0.0
    
    # Prepare labels
    if 'repertoire' in data[0]:
        labels = np.array([item.get('repertoire') for item in data])
    else:
        labels = np.array([item.get('prediction') for item in data])
    
    # Prepare embeddings
    X = np.array([item['embeddings'] for item in data])
    X = np.nan_to_num(X)

    labels_arr = np.asarray(labels).astype(str)
    unique_labels = np.unique(labels_arr).astype(str).tolist()

    if X.shape[1] > 1:      
        pred = labels_arr != 'Unknown'
        
        # UMAP dimensionality reduction
        import umap
        reducer = umap.UMAP(random_state=random_state)
        
        # # ParametricUMAP dimensionality reduction
        # from umap.parametric_umap import ParametricUMAP
        # from tensorflow import keras
        
        # n_components = 2
        # dims = (embeddings.shape[1],)
        # encoder = keras.Sequential([
        #     keras.layers.Input(shape=dims),
        #     keras.layers.Flatten(),
        #     keras.layers.Dense(units=512, activation="relu"),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.Dropout(0.3),
        #     keras.layers.Dense(units=256, activation="relu"),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.Dropout(0.3),
        #     keras.layers.Dense(units=128, activation="relu"),
        #     keras.layers.BatchNormalization(),
        #     keras.layers.Dropout(0.3),
        #     keras.layers.Dense(units=n_components, name="z")
        # ])

        # decoder = keras.Sequential([
        #         keras.layers.Input(shape=(n_components,)),
        #         keras.layers.BatchNormalization(),
        #         keras.layers.Dropout(0.3),
        #         keras.layers.Dense(units=128, activation="relu"),
        #         keras.layers.BatchNormalization(),
        #         keras.layers.Dropout(0.3),
        #         keras.layers.Dense(units=256, activation="relu"),
        #         keras.layers.BatchNormalization(),
        #         keras.layers.Dropout(0.3),
        #         keras.layers.Dense(units=512, activation="relu"),
        #         keras.layers.BatchNormalization(),
        #         keras.layers.Dropout(0.3),
        #         keras.layers.Dense(units=np.prod(dims), name="recon", activation=None),
        #         keras.layers.Reshape(dims),
        # ])

        # keras_fit_kwargs = {"callbacks": [
        #     keras.callbacks.EarlyStopping(
        #         monitor='loss',
        #         min_delta=10**-2,
        #         patience=10,
        #         verbose=1,
        #     )
        # ]}

        # reducer = ParametricUMAP(
        #                         encoder=encoder, 
        #                         decoder=decoder,
        #                         parametric_reconstruction=True,
        #                         n_components=n_components, 
        #                         metric=metric, 
        #                         random_state=self.random_state, 
        #                         verbose=self.verbose,
        #                         dims=dims, 
        #                         keras_fit_kwargs=keras_fit_kwargs,
        #                         )
        
        # reducer.n_training_epochs = 1  

        y = np.full(labels_arr.shape, -1, dtype=int)

        if pred.any():
            pred_labels = labels_arr[pred]
            _, y_pred = np.unique(pred_labels, return_inverse=True)
            y[pred] = y_pred
            
        X = reducer.fit_transform(X, y=y)

        # if pred.any():
        #     # Fit only on predicted samples (i.e. not 'Unknown')
        #     X_pred = X[pred]
        #     pred_labels = labels_arr[pred]
        #     _, y_pred = np.unique(pred_labels, return_inverse=True)
        #     reducer.fit(X_pred, y=y_pred)
        #     # Transform all samples with the autoencoder
        #     X = reducer.transform(X)
        # else:
        #     # If nothing is predicted, fall back to fitting on all data
        #     X = reducer.fit_transform(X)
        
    return X, labels_arr, unique_labels

def run_bokeh_server(data, audio_filter_chain=None, min_confidence=0.90, frame_width=1400, frame_height=600, point_size=5, port=8080, random_state=2025):
        
    # Supervised dimensionality reduction with UMAP
    X, labels_arr, unique_labels = reducer(data, min_confidence=min_confidence, random_state=random_state)

    def modify_doc(doc):
        x, y = X[:, 0], X[:, 1]
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

        try:
            colors = Category20[len(unique_labels)]
        except:
            colors = Category20[20]

        palette = colors if len(colors) >= len(unique_labels) else (colors * ((len(unique_labels) // len(colors)) + 1))
        color_list = [palette[label_to_idx[lbl]] for lbl in labels_arr]
        size_list = [point_size if lbl == 'Unknown' else point_size * 3 for lbl in labels_arr]

        source = ColumnDataSource(data=dict(
            x=x, y=y, label=labels_arr, audio=[""] * len(data), 
            path=[emb['path'] for emb in data],
            tmin=[emb['tmin'] for emb in data],
            tmax=[emb['tmax'] for emb in data],
            fmin=[emb['fmin'] for emb in data],
            fmax=[emb['fmax'] for emb in data],
            rate=[emb['rate'] for emb in data],
            color=color_list, size=size_list
        ))

        # Create the figure with TapTool enabled and box_select active by default
        p = figure(
            frame_width=frame_width,
            frame_height=frame_height,
            title=f"Ecological Clustering with Analysis of Sound Events (ECOCASE): {len(unique_labels)} clusters",
            tools="pan,wheel_zoom,reset,tap,box_select,lasso_select",
            active_scroll='wheel_zoom',
            active_drag='box_select'
        )

        # Scatter plot
        p.scatter('x', 'y', size='size', color='color', source=source, legend_field='label', alpha=0.95)
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"

        # TapTool setup
        taptool = p.select(type=TapTool)
        taptool.callback = CustomJS(args=dict(source=source), code="""
            const selected_index = source.selected.indices[0];
            console.log("Selected index:", selected_index);
        """)

        audio_div = Div(text="")
        spectrogram_div = Div(text="")
        status_div = Div(text="")  # status messages for downloads

        def show_rois():
            selected = source.selected.indices
            if not selected:
                return

            combined_html = ''
            row_open = False

            for i, idx in enumerate(selected):
                if i % 3 == 0:
                    if row_open:
                        combined_html += '</div>'
                    combined_html += '<div style="display: flex; justify-content: center; margin-bottom: 20px;">'
                    row_open = True

                path = source.data['path'][idx]
                tmin = source.data['tmin'][idx]
                tmax = source.data['tmax'][idx]
                fmin = source.data['fmin'][idx]
                fmax = source.data['fmax'][idx]
                rate = source.data['rate'][idx]

                # Find the original item in 'data' that matches this ROI and grab its "roi" key
                for item in data:
                    if (item.get('path') == path and
                        float(item.get('tmin')) == float(tmin) and
                        float(item.get('tmax')) == float(tmax) and
                        float(item.get('fmin')) == float(fmin) and
                        float(item.get('fmax')) == float(fmax)):
                        roi_audio = item['roi']

                # Write WAV bytes for both audio playback and for piping into ffmpeg
                buf = io.BytesIO()
                wavfile.write(buf, rate, roi_audio)
                buf.seek(0)
                wav_bytes = buf.read()
                b64 = base64.b64encode(wav_bytes).decode('ascii')
                audio_result = f"data:audio/wav;base64,{b64}"

                # Use ffmpeg to produce a spectrogram image by piping the WAV bytes to stdin
                cmd = [
                    "ffmpeg",
                    "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                    "-lavfi", "showspectrumpic=s=300x200:legend=enabled:color=channel:scale=log",
                    "-frames:v", "1", "-f", "image2", "pipe:1",
                ]
                proc = subprocess.run(cmd, input=wav_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                img_b64 = base64.b64encode(proc.stdout).decode("ascii")
                spectrogram_result = f'<img src="data:image/png;base64,{img_b64}" />'

                # Show the file path (and an optional clickable file:// link)
                path_html = f'<div style="font-size:12px; color:#444; margin-top:6px;">File path: {path}</a></div>'

                combined_html += f'''
                    <div style="display: flex; flex-direction: column; align-items: center; margin: 10px;">
                        {spectrogram_result}
                        <audio controls src="{audio_result}"></audio>
                        {path_html}
                    </div>
                '''

            if row_open:
                combined_html += '</div>'  # Close final row

            spectrogram_div.text = combined_html
            audio_div.text = ""  # Clear audio_div since it's merged

        def download_rois():
            inds = list(source.selected.indices)
            if not inds:
                def upd_empty():
                    status_div.text = "No selection to download."
                doc.add_next_tick_callback(upd_empty)
                return

            status_div.text = f"Starting download of {len(inds)} ROI(s)..."

            saved = []
            for idx in inds:
                try:
                    path = source.data['path'][idx]
                    tmin = source.data['tmin'][idx]
                    tmax = source.data['tmax'][idx]
                    fmin = source.data['fmin'][idx]
                    fmax = source.data['fmax'][idx]
                    rate = source.data['rate'][idx]

                    _, outpath = extract_roi_audio([path, tmin, tmax, fmin, fmax, audio_filter_chain, True, rate])
                    saved.append(outpath)

                except Exception as e:
                    print(f"Error running ffmpeg for idx {idx}: {e}")
                    saved.append(f"ERROR_saving_idx_{idx}: {e}")

                def update_status():
                    status_div.text = "Saved ROIs: " + ", ".join(saved)

            # update status on the document's next tick
            doc.add_next_tick_callback(update_status)

        # Reset button: clears selection, spectrograms, audio players and status text.
        def reset_all():
            def do_reset():
                # Clear selection if source exists
                try:
                    source.selected.indices = []
                except Exception:
                    pass
                # Clear displayed HTML & status if divs exist
                try:
                    spectrogram_div.text = ""
                    audio_div.text = ""
                    status_div.text = ""
                except Exception:
                    pass
            doc.add_next_tick_callback(do_reset)


        # Single button
        roi_button = Button(label="Show selected ROIs", button_type="primary")
        roi_button.on_click(show_rois)

        # Download button using ffmpeg to save selected ROIs to "rois" folder.
        download_button = Button(label="Download selected ROIs", button_type="success")
        download_button.on_click(download_rois)

        reset_button = Button(label="Reset", button_type="warning")
        reset_button.on_click(reset_all)

        # Attach the reset button to the buttons row.
        # This callback will run on the next tick when centered_buttons exists.
        def attach_reset_button():
            try:
                centered_buttons.children.append(reset_button)
            except Exception:
                pass

        doc.add_next_tick_callback(attach_reset_button)

        # Sizing
        p.sizing_mode = "stretch_width"
        audio_div.sizing_mode = "stretch_width"
        spectrogram_div.sizing_mode = "stretch_width"
        status_div.sizing_mode = "stretch_width"

        # Layout
        centered_plot = row(p, align="center")
        centered_buttons = row(roi_button, download_button, align="center")
        centered_audio = row(audio_div, align="center")
        centered_spectrogram = row(spectrogram_div, align="center")
        centered_status = row(status_div, align="center")

        layout = column(
            centered_plot,
            centered_buttons,
            centered_status,
            centered_audio,
            centered_spectrogram,
            sizing_mode="stretch_width",
        )

        doc.add_root(layout)

    # Start Bokeh server and open the browser to the app URL
    server = Server({'/': modify_doc}, port=port, allow_websocket_origin=[f"localhost:{port}"])
    server.start()
    url = f"http://localhost:{port}"
    print(f"Opening interactive display at {url}")

    try:
        webbrowser.open_new_tab(url)
    except Exception:
        pass

    server.io_loop.start()


# if __name__ == "__main__":
    
#     data = np.load('data.npy', allow_pickle=True)
#     audio_filter_chain = ",".join([
#             # Resample audio file
#             f"aresample=resampler=soxr:out_sample_rate=48000",
#             # Convert to mono
#             f"aformat=channel_layouts=mono",
#         ])
    
#     run_bokeh_server(data, audio_filter_chain, min_confidence=0.75, frame_width=1400, frame_height=600, point_size=5, port=8080, random_state=2025)
