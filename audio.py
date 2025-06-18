import os
import subprocess
import numpy as np
import librosa
import cv2
import csv
from concurrent.futures import ThreadPoolExecutor

from birdnet_analyzer import config as cfg


def load_audio(path, sr=cfg.SAMPLE_RATE, fmin=cfg.BANDPASS_FMIN, fmax=cfg.BANDPASS_FMAX, offset=0.0, duration=None, 
               dynaudnorm=True, denoise=True, noise_profile=[0.0,1.0], noise_floor=-20, order=5, output_filename=None):
    
    # Build FFmpeg command
    cmd = ["ffmpeg", "-y", "-hide_banner", "-ss", str(offset), "-i", str(path)]
    
    # Build filter chain for FFmpeg
    filters = []

    if dynaudnorm:
        # Dynamic audio normalization with alternative boundary mode
        filters.append("dynaudnorm=b=enabled")

    if fmin is not None:
        filters.append(f"highpass=f={fmin:.1f}")
    if fmax is not None:
        filters.append(f"lowpass=f={fmax:.1f}")

    if denoise:
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

    if output_filename is not None:
        # Write audio to disk
        cmd += [output_filename]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error writing audio file {output_filename}: {e}")
        return None

    # Load audio into memory
    cmd += ["-f", "f32le", "-ar", str(sr), "-ac", "1", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = proc.communicate()
    y = np.frombuffer(out, dtype='float32')
    
    return y, sr


def detect_rois(path, n_fft=2048, hop_length=1024, morph_preset="medium", min_time_px=20, min_freq_px=20,
                denoise=True, plot=False):
    """
    Detects ROIs and returns their time and frequency bounds.
    Returns a list of dicts with bbox, t_start, t_end, fmin, fmax.
    """

    # Load audio with FFmpeg
    y, sr = load_audio(path, denoise=denoise)

    # Compute spectrogram with librosa
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # Convert frequency axis to log scale for improving ROIs detection
    n_bins = S_db.shape[0]
    n_frames = S_db.shape[1]
    # Define log-spaced frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    log_freqs = np.geomspace(freqs[1], freqs[-1], num=n_bins) # avoid 0 Hz
    # Interpolate S_db onto log-frequency bins
    S_db_log = np.empty((n_bins, n_frames), dtype=np.float32)
    for t in range(n_frames):
        S_db_log[:, t] = np.interp(log_freqs, freqs, S_db[:, t])
    
    # Enhance contrast before thresholding
    spectrogram_img = cv2.normalize(S_db_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Set morphology preset
    kernel_shape = (15, 3) if morph_preset == "medium" else (7, 3)
    dilation_iter = 2 if morph_preset == "medium" else 1
    closing = morph_preset in ["medium", "aggressive"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)

    # Get binary mask with Otsu's threshold
    _, mask = cv2.threshold(spectrogram_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations
    dilated = cv2.dilate(mask, kernel, iterations=dilation_iter)
    if closing:
        dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Find the contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Group ROIs according to minimal time and frequency pixel threshold
    rois = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
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

    # Plot spectrogram with ROI contours if requested
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


def split_audio(path, rois, min_duration=cfg.SIG_MINLEN, max_duration=cfg.SIG_LENGTH,  
                denoise=True, output_folder=None, csv_filename='rois.csv', download=False):
    
    # Load audio with FFmpeg
    y, _ = load_audio(path, denoise=denoise)
    
    results = []

    if download:
        if output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)
            csv_path = os.path.join(output_folder, csv_filename)
        else:
            csv_path = csv_filename
        f_csv = open(csv_path, "w", newline="")
        writer = csv.writer(f_csv)
        writer.writerow(["ROI Index", "Start Time (s)", "End Time (s)", "Fmin (Hz)", "Fmax (Hz)", "Filename"])

    with ThreadPoolExecutor(max_workers=cfg.CPU_THREADS) as executor:
        futures = []
        for idx, group in enumerate(rois):
            t_start, t_end, f_min, f_max = group["bbox"]
            # Check minimum duration of ROI
            dur = t_end - t_start
            if dur < min_duration:
                continue
            # Split ROI according to maximum duration 
            segments = int(dur // max_duration)

            for i in range(segments):
                seg_start = round(t_start + i * max_duration, 3)
                output_name = f"roi_{idx + 1}_{i + 1}.mp3"
                if output_folder is not None:
                    save_path = os.path.join(output_folder, output_name) if download else None
                else:
                    save_path = output_name if download else None

                futures.append(executor.submit(
                    load_audio,
                    path=path,
                    offset=seg_start,
                    duration=max_duration,
                    fmin=f_min,
                    fmax=f_max,
                    denoise=denoise,
                    output_filename=save_path,
                ))

                results.append({
                    "roi_index": idx + 1,
                    "segment_index": i + 1,
                    "start": seg_start,
                    "end": seg_start + max_duration,
                    "fmin": f_min,
                    "fmax": f_max,
                    "filename": path,
                    "raw_audio": None  # Placeholder, will be filled below
                })

                if download:
                    writer.writerow([
                        idx + 1, seg_start, seg_start + max_duration,
                        f_min, f_max, output_name
                    ])

                raw_audio = [f.result() for f in futures]
                for i, sig in enumerate(raw_audio):
                    if sig is not None:
                        y, _ = sig
                        results[i]["raw_audio"] = y

    if download:
        f_csv.close()

    return results
                

# path = "test_data/SMA04635_20220318_182700.wav"  
# # Detect regions of interest (ROIs)
# rois = detect_rois(path, morph_preset='medium', plot=True)
# # Split audio with ROIs
# sig_splits = split_audio(path, rois, csv_filename="rois.csv", output_folder="rois", download=True)
# print(sig_splits)
