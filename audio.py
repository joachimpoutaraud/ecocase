import os
import subprocess
import numpy as np
import librosa
import cv2
import csv
from concurrent.futures import ThreadPoolExecutor

from birdnet_analyzer import config as cfg


def load_audio(path, sr=cfg.SAMPLE_RATE, fmin=cfg.BANDPASS_FMIN, fmax=cfg.BANDPASS_FMAX, offset=0.0, duration=None, 
               denoise=True, noise_floor=-30, sample_noise=0.5, order=5, output_filename=None):
    
    # Build FFmpeg command
    cmd = ["ffmpeg", "-y", "-hide_banner", "-ss", str(offset), "-i", str(path)]
    
    # Build filter chain for FFmpeg
    filters = []

    if fmin is not None:
        filters.append(f"highpass=f={fmin:.1f}")
    if fmax is not None:
        filters.append(f"lowpass=f={fmax:.1f}")

    if denoise:
        filters += [
            "asendcmd=0.0 afftdn sn start",
            f"asendcmd={sample_noise} afftdn sn stop",
            *[f"afftdn=nf={noise_floor}:track_residual=enabled" for _ in range(order)]
        ]
    af = ",".join(filters)
    
    if af:
        cmd += ["-af", af]
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


def split_audio(path, rois, min_duration=cfg.SIG_MINLEN, max_duration=cfg.SIG_LENGTH, n_fft=2048, hop_length=1024, 
                denoise=True, output_folder=None, csv_filename='rois.csv', download=False):
    
    # Load audio with FFmpeg
    y, sr = load_audio(path, denoise=denoise)
    
    frame_to_time = hop_length / sr
    bin_to_freq = sr / n_fft
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

    with ThreadPoolExecutor() as executor:
        futures = []
        for idx, group in enumerate(rois):
            x, y, w, h = group["bbox"]
            t_start = x * frame_to_time
            t_end = (x + w) * frame_to_time
            dur = t_end - t_start
            if dur < min_duration:
                continue
            f_min = round(y * bin_to_freq, 3)
            f_max = round((y + h) * bin_to_freq, 3)
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
                    "f_min": f_min,
                    "f_max": f_max,
                    "filename": output_name,
                    "raw_audio": None  # Placeholder, will be filled below
                })

                if download:
                    writer.writerow([
                        idx + 1, seg_start, seg_start + max_duration,
                        f_min, f_max, output_name
                    ])

        raw_audio = [f.result() for f in futures]
        for i, (y, sr) in enumerate(raw_audio):
            results[i]["raw_audio"] = y

    if download:
        f_csv.close()

    return results


def detect_rois(path, n_fft=2048, hop_length=1024, morph_preset="medium", time_thresh=20, denoise=True):

    # Load audio with FFmpeg
    y, sr = load_audio(path, denoise=denoise)

    # Compute log-spectrogram with librosa
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    spectrogram_img = cv2.normalize(S_db, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Set morphology preset
    kernel_shape = (15, 3) if morph_preset == "medium" else (7, 3)
    dilation_iter = 2 if morph_preset == "medium" else 1
    closing = morph_preset in ["medium", "aggressive"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)

    _, mask = cv2.threshold(spectrogram_img, 150, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(mask, kernel, iterations=dilation_iter)
    if closing:
        dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Find the contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Group the contours according to time threshold
    groups = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        for g in groups:
            gx, gy, gw, gh = g["bbox"]
            if abs(x - (gx + gw)) <= time_thresh:
                new_x = min(x, gx)
                new_w = max(x + w, gx + gw) - new_x
                g["bbox"] = (new_x, min(y, gy), new_w, max(y + h, gy + gh) - min(y, gy))
                g["contours"].append(cnt)
                break
        else:
            groups.append({"bbox": (x, y, w, h), "contours": [cnt]})

    return groups
                

path = "wat.mp3"  
# Detect regions of interest (ROIs)
rois = detect_rois(path, morph_preset='medium')
# Split audio with ROIs
sig_splits = split_audio(path, rois, csv_filename="rois.csv")
print(sig_splits)