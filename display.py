import os, subprocess
import io, base64
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import webbrowser

import umap

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Button, Div, CustomJS, TapTool
from bokeh.layouts import column, row
from bokeh.palettes import Category20
from bokeh.server.server import Server

import warnings
warnings.filterwarnings("ignore")

def display_rois(embeddings, rate=16000, noise_reduction=-60, frame_width=1000, frame_height=600, point_size=5, port=8080):

    embeddings = [item for sublist in embeddings for item in sublist]  # Flatten the embedding list

    try:
        labels = np.array([item.get('prediction') for item in embeddings])
    except Exception:
        pass

    def render_roi_base64(path, tmin, tmax, fmin, fmax, rate, noise_reduction, mode='audio'):
        """
        Read a region-of-interest (ROI) from `path` (wav file) between tmin and tmax (seconds)
        and return either an audio data URI or a spectrogram <img/> HTML string.

        mode: 'audio' or 'spectrogram'
        """
        # Read file
        audio_filter = ",".join([
            # keep sample rate stable and convert to mono
            f"aresample=resampler=soxr:out_sample_rate={rate},aformat=channel_layouts=mono",
            f"firequalizer=gain='if(between(f,{int(fmin)},{int(fmax)}),0,{noise_reduction})'",
            # normalize dynamics so embeddings are consistent across files/ROIs
            "dynaudnorm=b=enabled",
        ])

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-v", "error",
            "-ss", str(tmin), "-t", str(tmax - tmin), "-i", path,
            "-filter_complex", audio_filter,
            "-f", "s16le", "pipe:1"
        ]
        # Run FFmpeg and read audio data
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            audio_bytes, error = process.communicate()
            roi_audio = np.frombuffer(audio_bytes, dtype=np.int16)

        result_audio = None
        result_img = None

        if mode == 'audio':
            buf = io.BytesIO()
            wavfile.write(buf, rate, roi_audio)
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode('ascii')
            result_audio = f"data:audio/wav;base64,{b64}"

            return result_audio

        if mode == 'spectrogram':
            # compute spectrogram and render to PNG (grayscale)
            f, t, Sxx = spectrogram(roi_audio, fs=rate)
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
                masked_labels = labels_arr[mask]
                unique_labels, y_masked = np.unique(masked_labels, return_inverse=True)
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
            fmin=[emb['fmin'] for emb in embeddings],
            fmax=[emb['fmax'] for emb in embeddings],
            color=color_list, size=size_list
        ))

        # Create the figure with TapTool enabled and box_select active by default
        p = figure(
            frame_width=frame_width,
            frame_height=frame_height,
            title="Interactive scatter plot",
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
                if i % 2 == 0:
                    if row_open:
                        combined_html += '</div>'
                    combined_html += '<div style="display: flex; justify-content: center; margin-bottom: 20px;">'
                    row_open = True

                path = source.data['path'][idx]
                tmin = source.data['tmin'][idx]
                tmax = source.data['tmax'][idx]
                fmin = source.data['fmin'][idx]
                fmax = source.data['fmax'][idx]

                spectrogram_result = render_roi_base64(path, tmin, tmax, fmin, fmax, rate, noise_reduction, mode='spectrogram')
                audio_result = render_roi_base64(path, tmin, tmax, fmin, fmax, rate, noise_reduction, mode='audio')

                combined_html += f'''
                    <div style="display: flex; flex-direction: column; align-items: center; margin: 10px;">
                    {spectrogram_result}
                    <audio controls src="{audio_result}"></audio>
                    </div>
                '''

            if row_open:
                combined_html += '</div>'  # Close final row

            spectrogram_div.text = combined_html
            audio_div.text = ""  # Clear audio_div since it's merged

        # Single button
        roi_button = Button(label="Show selected ROIs", button_type="primary")
        roi_button.on_click(show_rois)

        def download_rois():
            inds = list(source.selected.indices)
            if not inds:
                def upd_empty():
                    status_div.text = "No selection to download."
                doc.add_next_tick_callback(upd_empty)
                return

            os.makedirs("rois", exist_ok=True)
            status_div.text = f"Starting download of {len(inds)} ROI(s)..."

            saved = []
            for idx in inds:
                try:
                    path = source.data['path'][idx]
                    tmin = source.data['tmin'][idx]
                    tmax = source.data['tmax'][idx]
                    fmin = source.data['fmin'][idx]
                    fmax = source.data['fmax'][idx]

                    duration = max(0.001, float(tmax) - float(tmin))
                    base = os.path.splitext(os.path.basename(path))[0]
                    fname = f"{base}_t_{tmin:.3f}_{tmax:.3f}_f_{fmin:.3f}_{fmax:.3f}.wav"
                    outpath = os.path.join("rois", fname)

                    # Example ffmpeg command: trim, convert to mono and 48kHz
                    cmd = [
                    "ffmpeg", "-y", "-hide_banner", "-v", "error",
                    "-ss", str(tmin), "-t", str(duration),
                    "-i", path,
                    "-ac", "1", "-ar", "48000",
                    outpath
                    ]

                    subprocess.run(cmd, check=True)
                    saved.append(outpath)
                except Exception as e:
                    print(f"Error running ffmpeg for idx {idx}: {e}")
                    saved.append(f"ERROR_saving_idx_{idx}: {e}")

                def update_status():
                    status_div.text = "Saved ROIs: " + ", ".join(saved)

            # update status on the document's next tick
            doc.add_next_tick_callback(update_status)

        # Download button using ffmpeg to save selected ROIs to "rois" folder.
        download_button = Button(label="Download selected ROIs", button_type="success")
        download_button.on_click(download_rois)

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
        # best-effort to open browser; ignore failures and continue to start server loop
        pass

    server.io_loop.start()






