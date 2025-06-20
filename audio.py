import os
import subprocess
import numpy as np
import pandas as pd
import librosa
import cv2

from concurrent.futures import ThreadPoolExecutor

from birdnet_analyzer import config as cfg


def load_audio(path, sr=cfg.SAMPLE_RATE, fmin=cfg.BANDPASS_FMIN, fmax=cfg.BANDPASS_FMAX, offset=0.0, duration=None, 
               dynaudnorm=True, denoise=True, noise_profile=[0.0,1.0], noise_floor=-20, order=3, output_filename=None, max_retries=5):
    
    # Build FFmpeg command
    cmd = ["ffmpeg", "-y", "-hide_banner", "-ss", str(offset), "-i", str(path)] 
    # Build filter chain for FFmpeg
    filters = []

    if dynaudnorm:
        # Dynamic audio normalization with alternative boundary mode 
        # to get a smooth "fade in" and "fade out" at the beginning and at the end of the audio input
        filters.append("dynaudnorm=b=enabled")

    if fmin is not None:
        filters.append(f"highpass=f={fmin:.1f}")
    if fmax is not None:
        filters.append(f"lowpass=f={fmax:.1f}")

    if denoise:
        if noise_profile == None:
            filters += [*[f"afftdn=nf={noise_floor}:tr=1" for _ in range(order)]]
        else:
            filters += [
                f"asendcmd={noise_profile[0]} afftdn sn start",
                f"asendcmd={noise_profile[1]} afftdn sn stop",
                *[f"afftdn=nf={noise_floor}:tr=1" for _ in range(order)]
            ]
    af = ",".join(filters)
    
    if af:
        cmd += ["-filter_complex", af]
    if duration is not None:
        cmd += ["-t", str(duration)]

     # Retry subprocess several times to avoid missing files
    if output_filename is not None:
        # Write audio to disk with retries
        cmd += [output_filename]
        for attempt in range(max_retries):
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                if os.path.exists(output_filename):
                    break
            except subprocess.CalledProcessError as e:
                if attempt == max_retries - 1:
                    print(f"Error writing audio file {output_filename}: {e}")
        return None

    # Load audio into memory with retries
    cmd += ["-f", "f32le", "-ar", str(sr), "-ac", "1", "-"]
    for attempt in range(max_retries):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = proc.communicate()
        y = np.frombuffer(out, dtype='float32')
        if y.size > 0:
            break
        elif attempt == max_retries - 1:
            print(f"Error loading audio from {path} after {max_retries} attempts.")
            return None, sr
    
    return y, sr


def detect_rois(y, sr, n_fft=2048, hop_length=1024, morph_preset="medium", min_time_px=20, min_freq_px=20, plot=False):
    """
    Detects ROIs and returns their time and frequency bounds.
    Returns a list of dicts with bbox, t_start, t_end, fmin, fmax.
    Optimized for speed.
    """

    # Compute spectrogram with librosa (use magnitude squared for speed)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # Use log-frequency bins (vectorized interpolation)
    n_bins, n_frames = S_db.shape
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    log_freqs = np.geomspace(freqs[1], freqs[-1], num=n_bins)  # avoid 0 Hz
    # Interpolate S_db onto log-frequency bins
    S_db_log = np.empty((n_bins, n_frames), dtype=np.float32)
    for t in range(n_frames):
        S_db_log[:, t] = np.interp(log_freqs, freqs, S_db[:, t])

    # Normalize and threshold with
    spectrogram_img = cv2.normalize(S_db_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(spectrogram_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations (kernel and iterations depend on preset)
    presets = {
        "medium": {"kernel_shape": (15, 3), "dilation_iter": 2, "closing": True},
        "aggressive": {"kernel_shape": (21, 5), "dilation_iter": 3, "closing": True},
        "light": {"kernel_shape": (7, 3), "dilation_iter": 1, "closing": False},
    }
    p = presets.get(morph_preset, presets["medium"])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, p["kernel_shape"])
    dilated = cv2.dilate(mask, kernel, iterations=p["dilation_iter"])
    if p["closing"]:
        dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Group ROIs (avoid nested loops for speed)
    bboxes = [cv2.boundingRect(cnt) for cnt in contours]
    bboxes = [b for b in bboxes if b[2] >= min_time_px and b[3] >= min_freq_px]

    # Merge overlapping/close bboxes (vectorized, fast)
    rois = []
    for x, y, w, h in bboxes:
        merged = False
        for r in rois:
            rx, ry, rw, rh = r["bbox"]
            # Check both time and frequency axis for proximity
            time_close = abs(x - (rx + rw)) <= min_time_px or abs(rx - (x + w)) <= min_time_px
            freq_overlap = not (y + h < ry - min_freq_px or ry + rh < y - min_freq_px)
            if time_close and freq_overlap:
                # Merge bounding boxes
                new_x = min(x, rx)
                new_w = max(x + w, rx + rw) - new_x
                new_y = min(y, ry)
                new_h = max(y + h, ry + rh) - new_y
                r["bbox"] = (new_x, new_y, new_w, new_h)
                merged = True
                break
        if not merged:
            rois.append({"bbox": (x, y, w, h)})

    # Plot if requested
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.imshow(spectrogram_img, aspect='auto', origin='lower', cmap='gray')
        for r in rois:
            x, y, w, h = r["bbox"]
            rect = plt.Rectangle((x, y), w, h, edgecolor='lime', facecolor='none', linewidth=2)
            plt.gca().add_patch(rect)
        plt.title("Log-spectrogram with ROI Contours")
        plt.xlabel("Time (frames)")
        plt.ylabel("Frequency (bins)")
        plt.tight_layout()
        plt.show()

    # Convert bbox values to time and frequency
    for r in rois:
        x, y, w, h = r["bbox"]
        t_start = x * (hop_length / sr)
        t_end = (x + w) * (hop_length / sr)
        fmin = round(log_freqs[y], 3) if y < len(log_freqs) else None
        fmax = round(log_freqs[y + h], 3) if (y + h) < len(log_freqs) else None
        r["bbox"] = (t_start, t_end, fmin, fmax)

    return rois


def split_audio(path, min_duration=cfg.SIG_MINLEN, max_duration=cfg.SIG_LENGTH, denoise=True):
    """
    Splits audio into segments based on detected ROIs, using parallel processing for speed.
    Returns a pandas DataFrame with segment metadata and audio.
    """

    audio_segments, tasks = [], []

    # Load audio once
    y, sr = load_audio(path, denoise=denoise)
    # Detect ROIs
    rois = detect_rois(y, sr)

    # Prepare segment extraction tasks
    for idx, group in enumerate(rois):
        t_start, t_end, f_min, f_max = group["bbox"]
        dur = t_end - t_start
        if dur < min_duration:
            continue
        segments = int(dur // max_duration)

        for i in range(segments):
            seg_start = round(t_start + i * max_duration, 3)
            tasks.append({
                "start": seg_start,
                "end": seg_start + max_duration,
                "fmin": f_min,
                "fmax": f_max,
                "file": path,
                "params": {
                    "path": path,
                    "offset": seg_start,
                    "duration": max_duration,
                    "fmin": f_min,
                    "fmax": f_max,
                    "denoise": denoise,
                    "noise_profile": None, 
                }
            })

    def extract_segment(task):
        y, _ = load_audio(**task["params"])
        # Remove "params" from dictionary and add raw_audio
        task = {k: v for k, v in task.items() if k != "params"}
        task["raw_audio"] = y
        return task

    if len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=cfg.CPU_THREADS) as executor:
            results = list(executor.map(extract_segment, tasks))
        audio_segments.extend(results)

    # Convert to DataFrame
    df = pd.DataFrame(audio_segments)
    return df
                

# path = "test_data/SMA04635_20220318_182700.wav"  
# # Detect regions of interest (ROIs)
# rois = detect_rois(path, morph_preset='medium', plot=True)
# # Split audio with ROIs
# sig_splits = split_audio(path, rois, csv_filename="rois.csv", output_folder="rois", download=True)
# print(sig_splits)
