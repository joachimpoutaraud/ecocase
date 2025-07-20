import os, sys, contextlib
import subprocess
import numpy as np
import pandas as pd

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try: yield
        finally: sys.stdout, sys.stderr = old_stdout, old_stderr

# ——— Configuration ——— #
min_conf = 0.75                   # Minimum confidence threshold
output_sr = 48000                 # Sample rate for audio
dataset_path = 'train/xc_dataset.csv'

# ——— Suppress stdout/stderr ——— #
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

# ——— Load audio with system ffmpeg into memory ——— #
def load_audio_with_ffmpeg(path, fmin=150, fmax=8000, normalize=True, denoise=True, noise_profile=[0.0,1.0], noise_floor=-20, order=3, sr=output_sr):
    # Create FFmpeg command
    cmd = ["ffmpeg", "-y", "-hide_banner", "-i", path] 
    # Build filter chain for FFmpeg
    filters = []
    
    if fmin is not None and fmax is not None:        
        filters.append(f"firequalizer=gain='if(lt(f,{fmin}),-INF,if(gt(f,{fmax}),-INF,0))'")
    if normalize:
        # Dynamic audio normalization with alternative boundary mode 
        # to get a smooth "fade in" and "fade out" at the beginning and at the end of the audio input
        filters.append("dynaudnorm=b=enabled")
    af = ",".join(filters)
    
    if af:
        cmd += ["-filter_complex", af]
    
    # Pipe float32 audio in mono  
    cmd += ["-f", "f32le", "-ar", str(sr), "-ac", "1", "-"]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    stdout_data, sterr_data = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg loading error: {sterr_data}")
    
    audio_buffer = np.frombuffer(stdout_data, dtype=np.float32)
    return audio_buffer

# ——— Save audio segment + generate spectrogram thumbnails ——— #
def save_audio_with_ffmpeg(url, start, end, label, identifier, output_dir="train", spectrogram=True, spectrogram_size="640x480", fscale="lin"):
    # Create target directory with segments folder
    label_path = os.path.join(output_dir, "train_data", label)
    os.makedirs(label_path, exist_ok=True)

    base_name = f"{identifier}_{start:.2f}-{end:.2f}"
    audio_out = os.path.join(label_path, f"{base_name}.wav")
    spec_out = os.path.join(label_path, f"{base_name}.png")
    
    audio_array = url.astype(np.float32)
    # Calculate start and end indices
    start_idx = int(start * output_sr)
    end_idx = int(end * output_sr)
    # Trim the array
    trimmed_array = audio_array[start_idx:end_idx]

    # Build FFmpeg command
    if spectrogram:
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-hide_banner',
            '-f', 'f32le',
            '-ar', str(output_sr),
            '-ac', '1',
            '-i', 'pipe:0',
            '-filter_complex',
            f"[0:a]asplit=2[out1][out2];"
            f'[out1]showspectrumpic=s={spectrogram_size}:legend=disabled:fscale={fscale}[outimg]',
            '-map', '[out2]', '-c:a', 'pcm_s16le', audio_out,
            '-map', '[outimg]', spec_out
        ]
    else:
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-hide_banner',
            '-f', 'f32le',
            '-ar', str(output_sr),
            '-ac', '1',
            '-i', 'pipe:0',
            '-c:a', 'pcm_s16le', audio_out
        ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    process.stdin.write(trimmed_array.tobytes())
    process.stdin.close()
    process.wait()

# ——— Process row: analyze, filter, save ——— #
def process_row(row, include_detections=True, include_embeddings=True, save_audio=True):
    try:
        from birdnetlib import RecordingBuffer
        from birdnetlib.analyzer import Analyzer
        
        # Pipe audio with FFmpeg
        audio_data = load_audio_with_ffmpeg(row['file'])
        
        with suppress_output():
            analyzer = Analyzer()
            recording = RecordingBuffer(analyzer, 
                                        audio_data, 
                                        output_sr,
                                        # min_conf=min_conf,
                                        # lat=row['lat'],
                                        # lon=row['lon']
                                        )
            if include_detections:
                # Analyze audio and make predictions
                recording.analyze()
            if include_embeddings:
                # Extract embeddings from BirdNET
                recording.extract_embeddings()

        results = []

        # Use full-audio embeddings if present
        emb_vector = None
        if include_embeddings:
            emb_vector = recording.embeddings[0]['embeddings']
            emb_dict = {f'emb{i+1:04d}': val for i, val in enumerate(emb_vector)}

        # Detections-based flow
        if include_detections:
            for i, det in enumerate(recording.detections):
                if det['confidence'] < min_conf:
                    continue

                if save_audio:
                    # if there are no identifier
                    save_audio_with_ffmpeg(
                        url=audio_data,
                        start=det['start_time'],
                        end=det['end_time'],
                        label=det['label'],
                        identifier=i
                    )

                result_entry = {
                    'file': row['file'],
                    'categories': det['common_name'],
                    'scientific_name': det['scientific_name'],
                    'label': det['label'],
                    'confidence': det['confidence'],
                    'start': det['start_time'],
                    'end': det['end_time']
                }

                if emb_vector:
                    result_entry.update(emb_dict)
                # Append the results to list
                results.append(result_entry)

        # Embedding-only with no detections
        elif emb_vector:
            result_entry = {
                'file': row['file'],
                'categories': row['categories'],
                'label': 'Unlabeled'
            }
            result_entry.update(emb_dict)
            results.append(result_entry)

        return results

    except Exception as e:
        print(f"💥 Failed to process {row['file']}: {e}")
        return []

if __name__ == "__main__":
    
    # Load the dataset
    from xcnet import XCNet
    xcnet = XCNet(save_path='Chloris')
    # Request dataset of the species of interest in Xeno-Canto 
    df = xcnet.dataset(csv_filename='xc_dataset.csv', download=True) 
    
    # ——— Load DataFrame and run in parallel ——— #
    # df = pd.read_csv(dataset_path, sep=';')
    
    # # Get list of subfolders in the dataset folder
    # from birdnet_analyzer import config as cfg
    # from pathlib import Path
    
    # dataset = "data/"
    # path = filter(lambda el: os.path.isdir(os.path.join(dataset, el)), os.listdir(dataset))
    # subfolders = sorted(path)

    # # If there are no subfolders, we assume it is an unlabeled dataset
    # if len(subfolders) == 0:
    #     # List all files with allowed extensions
    #     files = []
    #     for ext in cfg.ALLOWED_FILETYPES:
    #         files.extend(Path(dataset).rglob(f"*{ext}"))
    #     files = [str(f) for f in files]

    #     # Create dataframe for unlabeled dataset
    #     df = pd.DataFrame({'file': files, 'categories': ["Unlabeled"] * len(files)})
        
    # else:
    #     files = []
    #     categories = []
    #     for folder in subfolders:
    #         folder_path = os.path.join(dataset, folder)
    #         # List all files with allowed extensions
    #         for ext in cfg.ALLOWED_FILETYPES:
    #             for f in Path(folder_path).rglob(f"*{ext}"):
    #                 files.append(str(f))
    #                 categories.append(folder)

    #     # Create dataframe for training dataset
    #     df = pd.DataFrame({'file': files, 'categories': categories})
    
    record_list = df.to_dict(orient='records')
    detections = []

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_row, row) for row in record_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="⚡ Building dataset"):
            result = future.result()
            if result:
                detections.extend(result)
                
    # Save to CSV for inspection or downstream tasks
    detections_df = pd.DataFrame(detections)
    detections_df.to_csv('Chloris/chloris_chloris.csv', index=False, sep=';')