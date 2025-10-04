import numpy as np
import webbrowser

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Button, Div, CustomJS, TapTool
from bokeh.layouts import column, row
from bokeh.palettes import Category20
from bokeh.server.server import Server

from rois import extract_roi_audio

import warnings
warnings.filterwarnings("ignore")


def run_bokeh_server(data, audio_filter_chain=None, noise_reduction=-20, frame_width=1400, frame_height=600, point_size=5, port=8080, random_state=2025):
    # Prepare labels
    if 'repertoire' in data[0]:
        labels = np.array([item.get('repertoire') for item in data])
    else:
        labels = np.array([item.get('prediction') for item in data])

    def modify_doc(doc):
        # Prepare embeddings
        X = np.array([item['embeddings'] for item in data])
        X = np.nan_to_num(X)

        if X.shape[1] > 2:
            import umap
            reducer = umap.UMAP(random_state=random_state)
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

                spectrogram_result = extract_roi_audio([path, tmin, tmax, fmin, fmax, audio_filter_chain, noise_reduction, False, rate, 'spectrogram'])
                audio_result = extract_roi_audio([path, tmin, tmax, fmin, fmax, audio_filter_chain, noise_reduction, False, rate, 'audio'])

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

                    _, outpath = extract_roi_audio([path, tmin, tmax, fmin, fmax, audio_filter_chain, noise_reduction, True, rate, False])
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