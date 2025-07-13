import os
import subprocess
import numpy as np
import pandas as pd
import librosa
import cv2

from concurrent.futures import ThreadPoolExecutor

from birdnet_analyzer import config as cfg


def load_audio(path, sr=cfg.SAMPLE_RATE, fmin=cfg.BANDPASS_FMIN, fmax=cfg.BANDPASS_FMAX, offset=0.0, duration=None, 
               dynaudnorm=True, denoise=True, noise_profile=[0.0,1.0], noise_floor=-30, order=3, output_filename=None, max_retries=5):
        
    # Build FFmpeg command
    cmd = ["ffmpeg", "-y", "-hide_banner", "-ss", str(offset), "-i", str(path)] 
    # Build filter chain for FFmpeg
    filters = []

    # Use bandpass filter to keep only frequencies between fmin and fmax
    if fmin is not None and fmax is not None:        
        filters.append(f"firequalizer=gain='if(lt(f,{fmin}),-INF,if(gt(f,{fmax}),-INF,0))'")
        # filters.append(f"bandpass=f={((fmin + fmax) / 2):.1f}:width_type=h:w={(fmax - fmin):.1f}")
        # filters.append(f"highpass=f={fmin:.1f}")
        # filters.append(f"lowpass=f={fmax:.1f}")

    if dynaudnorm:
        filters.append("dynaudnorm=b=enabled")

    if denoise:
        if noise_profile is None:
            filters += [*[f"afftdn=nf={noise_floor}:tr=1" for _ in range(order)]]
        else:
            filters += [
                f"asendcmd={noise_profile[0]} afftdn sn start",
                f"asendcmd={noise_profile[1]} afftdn sn stop",
                *[f"afftdn=nf={noise_floor}:tr=1" for _ in range(order)],
                # f"afwtdn=sigma=1:wavet='bl3':percent=100:levels=12:profile=1:adaptive=1"
            ]
    # Join filter chain with commas
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
    for attempt in range(max_retries):
        cmd_mem = cmd + ["-f", "f32le", "-ar", str(sr), "-ac", "1", "-"]
        proc = subprocess.Popen(cmd_mem, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = proc.communicate()
        y = np.frombuffer(out, dtype='float32')
        if y.size > 0:
            break
        elif attempt == max_retries - 1:
            print(f"Error loading audio from {path} after {max_retries} attempts.")
            return None, sr
    
    return np.array(y, dtype='float32'), sr


def detect_rois(y, sr, fmin=cfg.BANDPASS_FMIN, fmax=cfg.BANDPASS_FMAX, n_fft=2048, hop_length=512, morph_preset="medium", min_time_px=20, min_freq_px=20, plot=False):
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
    # Filter spectrogram
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    S_db_filtered = S_db[freq_mask, :]
    filtered_freqs = freqs[freq_mask]

    # Use log-frequency bins within filtered range
    n_bins, n_frames = S_db_filtered.shape
    log_freqs = np.geomspace(filtered_freqs[0], filtered_freqs[-1], num=n_bins)
    
    # Interpolate S_db onto log-frequency bins
    S_db_log = np.empty((n_bins, n_frames), dtype=np.float32)
    for t in range(n_frames):
        S_db_log[:, t] = np.interp(log_freqs, freqs, S_db[:, t])
        
    # # Use log-frequency bins (vectorized interpolation)
    # n_bins, n_frames = S_db.shape
    # freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    # log_freqs = np.geomspace(freqs[1], freqs[-1], num=n_bins)  # avoid 0 Hz
    # # Interpolate S_db onto log-frequency bins
    # S_db_log = np.empty((n_bins, n_frames), dtype=np.float32)
    # for t in range(n_frames):
    #     S_db_log[:, t] = np.interp(log_freqs, freqs, S_db[:, t])

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
    # Group bboxes (avoid nested loops for speed)
    bboxes = [cv2.boundingRect(cnt) for cnt in contours]
    bboxes = [b for b in bboxes if b[2] >= min_time_px and b[3] >= min_freq_px]
    # Remove bboxes that are suspiously large
    image_h, image_w = spectrogram_img.shape
    bboxes = [b for b in bboxes if b[2] < image_w * 0.99 and b[3] < image_h * 0.99]

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
        plt.ylabel("Frequency (Hz)")
        # Set y-ticks to log frequencies
        ax = plt.gca()
        yticks = ax.get_yticks()
        yticks = yticks[yticks >= 0]
        yticks = yticks[yticks < len(log_freqs)]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{log_freqs[int(y)]:.0f}" for y in yticks])
        plt.tight_layout()
        plt.show()

    # Convert bbox values to time and frequency
    for r in rois:
        x, y, w, h = r["bbox"]
        tstart = x * (hop_length / sr)
        tend = (x + w) * (hop_length / sr)
        f_min = round(log_freqs[y], 3) if y < len(log_freqs) else fmin
        f_max = round(log_freqs[y + h], 3) if (y + h) < len(log_freqs) else fmax
        r["bbox"] = (tstart, tend, f_min, f_max)

    return rois

# def split_audio(path, config, denoise=True, min_duration=cfg.SIG_MINLEN, max_duration=cfg.SIG_LENGTH):
#     """
#     Splits audio into segments based on detected ROIs.
#     Applies bandpass filter to each segment and uses parallel processing.
#     Returns a pandas DataFrame with segment metadata and filtered audio.
#     """
    
#     # Set configuration parameters
#     cfg.set_config(config)

#     # Load full audio once
#     y, sr = load_audio(path, denoise=denoise)

#     # Detect ROIs
#     rois = detect_rois(y, sr)

#     if cfg.KEEP == 'BIGGEST':
#         fmins = [r["bbox"][2] for r in rois if r["bbox"][2] is not None]
#         fmaxs = [r["bbox"][3] for r in rois if r["bbox"][3] is not None]
#         if fmins and fmaxs:
#             lower_fmin = np.percentile(fmins, 10)
#             upper_fmax = np.percentile(fmaxs, 90)
#             rois = [
#                 r for r in rois
#                 if r["bbox"][2] is not None and r["bbox"][3] is not None and
#                    r["bbox"][3] >= lower_fmin and r["bbox"][2] <= upper_fmax
#             ]

#     tasks = []
#     for r in rois:
#         tstart, tend, fmin, fmax = r["bbox"]
#         dur = tend - tstart
#         if dur < min_duration:
#             continue

#         segments = int(np.ceil(dur / max_duration))
#         for i in range(segments):
#             seg_start = round(tstart + i * max_duration, 3)
#             seg_dur = min(max_duration, dur - i * max_duration)
#             if seg_dur < min_duration:
#                 continue
#             duration = max_duration if seg_dur >= max_duration else seg_dur
#             tasks.append({
#                 "start": seg_start,
#                 "end": seg_start + duration,
#                 "fmin": fmin,
#                 "fmax": fmax
#             })

#     from scipy.signal import lfilter, firwin
#     def fir_bandpass_filter(signal, sr, fmin, fmax, numtaps=1024):
#         taps = firwin(numtaps, [fmin, fmax], pass_zero=False, fs=sr)
#         return lfilter(taps, 1.0, signal)
    
#     # from scipy.signal import butter, sosfilt
#     # def butter_bandpass_sosfilter(signal, sr, fmin, fmax, order=3):
#     #     nyquist = 0.5 * sr
#     #     low = fmin / nyquist
#     #     high = fmax / nyquist
#     #     sos = butter(order, [low, high], btype='band', output='sos')
#     #     return sosfilt(sos, signal)

#     def extract_segment_from_loaded_audio(task, y, sr):
#         start_sample = int(task["start"] * sr)
#         end_sample = int(task["end"] * sr)
#         segment = y[start_sample:end_sample]

#         expected_len = int(sr * max_duration)
#         if len(segment) < expected_len:
#             segment = np.pad(segment, (0, expected_len - len(segment)), mode='constant')
#         # Filter loaded audio with FIR bandpass filter
#         filtered_segment = fir_bandpass_filter(segment, sr, task["fmin"], task["fmax"])

#         task["rois"] = filtered_segment
#         return task

#     audio_segments = []
#     if len(tasks) > 1:
#         with ThreadPoolExecutor(max_workers=cfg.CPU_THREADS) as executor:
#             results = list(executor.map(lambda task: extract_segment_from_loaded_audio(task, y, sr), tasks))
#         audio_segments.extend(results)

#     df = pd.DataFrame(audio_segments)
#     return df


def split_audio(path, config, denoise=True, min_duration=cfg.SIG_MINLEN, max_duration=cfg.SIG_LENGTH):
    """
    Splits audio into segments based on detected ROIs, using parallel processing for speed.
    Returns a pandas DataFrame with segment metadata and audio.
    """
    
    # Set configuration parameters
    cfg.set_config(config) 
    audio_segments, tasks = [], []

    # Load audio once
    y, sr = load_audio(path, denoise=denoise)
    # Detect ROIs
    rois = detect_rois(y, sr)

    # Prepare segment extraction tasks
    for r in rois:
        tstart, tend, fmin, fmax = r["bbox"]
        
        # Skip segments shorter than min_duration
        dur = tend - tstart
        if dur < min_duration:
            continue        
        
        # Split segment into max_duration chunks, pad last if needed
        segments = int(np.ceil(dur / max_duration))
        for i in range(segments):
            seg_start = round(tstart + i * max_duration, 3)
            seg_dur = min(max_duration, dur - i * max_duration)
            
            # Skip segments shorter than min_duration
            if seg_dur < min_duration:
                continue
            
            # For the last segment, set duration to seg_dur, else max_duration
            duration = max_duration if seg_dur >= max_duration else seg_dur
            tasks.append({
                "start": seg_start,
                "end": seg_start + duration,
                "fmin": fmin,
                "fmax": fmax,
                "file": path,
                "params": {
                    "path": path,
                    "sr": sr,
                    "fmin": fmin,
                    "fmax": fmax,
                    "offset": seg_start,
                    "duration": duration,
                    "denoise": False,
                    # # Use white noise as profile to preserve ROI information
                    # "noise_profile": None,
                }
            })
                         
        def extract_segment(task):
            y, _ = load_audio(**task["params"])
            # Pad audio if shorter than max_duration
            expected_len = int(sr * max_duration)
            if y is not None and len(y) < expected_len:
                y = np.pad(y, (0, expected_len - len(y)), mode='constant')
            # Remove "params" from dictionary and add raw_audio
            task = {k: v for k, v in task.items() if k != "params"}
            task["rois"] = y
            return task

    if len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=cfg.CPU_THREADS) as executor:
            results = list(executor.map(extract_segment, tasks))
        audio_segments.extend(results)

    # Convert to DataFrame
    df = pd.DataFrame(audio_segments)
    return df