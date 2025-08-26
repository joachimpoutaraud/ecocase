from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Button, Div, CustomJS, TapTool

from bokeh.layouts import column, row
from bokeh.palettes import Category20
from bokeh.server.server import Server
import numpy as np
import pandas as pd
import umap
import io
import base64

import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

def interactive_display(embeddings, labels, frame_width=1000, frame_height=600, point_size=5, port=5006):

    def render_roi_base64(path, tmin, tmax, mode='audio'):
        """
        Read a region-of-interest (ROI) from `path` (wav file) between tmin and tmax (seconds)
        and return either an audio data URI or a spectrogram <img/> HTML string.

        mode: 'audio' or 'spectrogram'
        """
        # Read file
        rate, data = wavfile.read(path)
        total_samples = data.shape[0]
        duration = total_samples / float(rate)

        # Clamp times
        tmin = max(0.0, min(tmin, duration))
        tmax = max(tmin, min(tmax, duration))

        start_sample = int(tmin * rate)
        end_sample = int(tmax * rate)
        if end_sample <= start_sample:
            end_sample = min(start_sample + 1, total_samples)

        roi = data[start_sample:end_sample]

        # If stereo, convert to mono
        if getattr(roi, "ndim", 1) > 1:
            roi = roi.mean(axis=1)

        result_audio = None
        result_img = None

        if mode == 'audio':
            # Ensure integer PCM for wav write: convert floats to int16
            roi_audio = roi
            if roi_audio.dtype.kind == 'f':
                # normalize if in -1..1 range, otherwise scale safely
                max_abs = np.max(np.abs(roi_audio)) if roi_audio.size else 1.0
                scale = 32767.0 / max(1e-9, max_abs)
                roi_audio = (roi_audio * scale).astype(np.int16)
            elif roi_audio.dtype.kind in ('i', 'u'):
                # Cast to int16 if larger than int16 to keep compatibility
                if roi_audio.dtype.itemsize > 2:
                    roi_audio = roi_audio.astype(np.int16)

            buf = io.BytesIO()
            wavfile.write(buf, rate, roi_audio)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode('ascii')
            result_audio = f"data:audio/wav;base64,{b64}"

            return result_audio

        if mode == 'spectrogram':
            # compute spectrogram and render to PNG (grayscale)
            f, t, Sxx = spectrogram(roi, fs=rate)
            # convert to dB safely (use power)
            Sxx_db = 10 * np.log10(np.abs(Sxx)**2 + 1e-10)

            plt.figure(figsize=(6, 4))
            plt.pcolormesh(t, f, Sxx_db, shading='auto', cmap='gray')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title('Spectrogram')
            buf_img = io.BytesIO()
            plt.savefig(buf_img, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            buf_img.seek(0)
            img_b64 = base64.b64encode(buf_img.read()).decode('ascii')
            result_img = f'<img src="data:image/png;base64,{img_b64}" />'

            return result_img
        
        else:
            raise ValueError("mode must be either 'audio' or 'spectrogram'")

    def modify_doc(doc):
        # Prepare embeddings
        X = np.array([item['embeddings'] for item in embeddings])
        X = np.nan_to_num(X)

        if X.shape[1] > 2:
            reducer = umap.UMAP(n_components=2, metric='cosine', random_state=2025, transform_seed=2025)
            labels_arr = np.asarray(labels).astype(str)
            mask = labels_arr != 'Unknown'
            y = np.full(labels_arr.shape, -1, dtype=int)
            if mask.any():
                y_masked = pd.factorize(labels_arr[mask])[0]
                y[mask] = y_masked
            X = reducer.fit_transform(X, y=y)

        labels_arr = np.array(labels).astype(str)
        unique_labels = np.unique(labels_arr).astype(str).tolist()
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
            x=x, y=y, label=labels_arr, audio=[""] * len(embeddings), 
            path=[emb['path'] for emb in embeddings],
            tmin=[emb['tmin'] for emb in embeddings],
            tmax=[emb['tmax'] for emb in embeddings],
            color=color_list, size=size_list
        ))

        # Create the figure with TapTool enabled and wheel zoom active by default
        p = figure(
            frame_width=frame_width,
            frame_height=frame_height,
            title="Interactive scatter plot",
            tools="pan,wheel_zoom,reset,tap,box_select,lasso_select",
            active_scroll='wheel_zoom'
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

        def make_on_click(mode):
            def _on_click():
                selected = source.selected.indices
                if not selected:
                    return

                if mode == 'audio':
                    audio_players_html = '<div style="text-align: center;">'
                    for idx in selected:
                        path = source.data['path'][idx]
                        tmin = source.data['tmin'][idx]
                        tmax = source.data['tmax'][idx]
                        result = render_roi_base64(path, tmin, tmax, mode='audio')
                        audio_players_html += f'<audio controls src="{result}"></audio><br/>'
                        audio_players_html += '</div>'

                    audio_div.text = audio_players_html
                    spectrogram_div.text = ""

                else:  # spectrogram
                    imgs_html = '<div style="text-align: center;">'
                    for idx in selected:
                        path = source.data['path'][idx]
                        tmin = source.data['tmin'][idx]
                        tmax = source.data['tmax'][idx]
                        result = render_roi_base64(path, tmin, tmax, mode='spectrogram')
                        imgs_html += result + "<br/>"
                        imgs_html += '</div>'

                    spectrogram_div.text = imgs_html
                    audio_div.text = ""
            return _on_click

        spectrogram_button = Button(label="Show spectrogram", button_type="primary")
        spectrogram_button.on_click(make_on_click('spectrogram'))

        audio_button = Button(label="Play audio", button_type="success")
        audio_button.on_click(make_on_click('audio'))

        p.sizing_mode = "stretch_width"
        audio_div.sizing_mode = "stretch_width"
        spectrogram_div.sizing_mode = "stretch_width"

        # Wrap each component in a row with align="center"
        centered_plot = row(p, align="center")
        centered_buttons = row(audio_button, spectrogram_button, align="center")
        centered_audio = row(audio_div, align="center")
        centered_spectrogram = row(spectrogram_div, align="center")

        # Final layout
        layout = column(
            centered_plot,
            centered_buttons,
            centered_audio,
            centered_spectrogram,
            sizing_mode="stretch_width",  
        )

        doc.add_root(layout)

    server = Server({'/': modify_doc}, port=port, allow_websocket_origin=[f"localhost:{port}"])
    server.start()
    print(f"Interactive display accessible at http://localhost:{port}")
    server.io_loop.start()






