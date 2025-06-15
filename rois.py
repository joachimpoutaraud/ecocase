import os
import subprocess
import numpy as np
import librosa
import cv2
import csv
from concurrent.futures import ThreadPoolExecutor


def ffmpeg_cmd(path, rate, output_filename=None, offset=0.0, duration=None,
               fmin=None, fmax=None, denoise=True, noise_floor=-30,
               sample_noise=0.5, order=5):
    
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
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return None

    # Load audio into memory
    cmd += ["-f", "f32le", "-ar", str(rate), "-ac", "1", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = proc.communicate()
    
    return np.frombuffer(out, dtype='float32')
    
def detect_contours(y, n_fft=2048, hop_length=1024, morph_preset="medium", time_thresh=20):
    # Compute spectrogram image with librosa
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

def segment_rois(path, rate, output_folder=None, csv_filename='rois.csv',
                n_fft=2048, hop_length=1024, min_duration=1.0, max_duration=3.0,
                morph_preset="medium", time_thresh=20, denoise=True, download=False):
    
    y = ffmpeg_cmd(path, rate=rate)
    # Detect contours of ROIs
    contours = detect_contours(y, n_fft=n_fft, hop_length=hop_length, morph_preset=morph_preset, time_thresh=time_thresh)
    
    frame_to_time = hop_length / rate
    bin_to_freq = rate / n_fft
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
        for idx, group in enumerate(contours):
            x, y, w, h = group["bbox"]
            t_start = x * frame_to_time
            t_end = (x + w) * frame_to_time
            dur = t_end - t_start
            if dur < min_duration:
                continue
            f_min = y * bin_to_freq
            f_max = (y + h) * bin_to_freq
            segments = int(dur // max_duration)

            for i in range(segments):
                seg_start = t_start + i * max_duration
                output_name = f"roi_{idx + 1}_{i + 1}.mp3"
                if output_folder is not None:
                    save_path = os.path.join(output_folder, output_name) if download else None
                else:
                    save_path = output_name if download else None

                futures.append(executor.submit(
                    ffmpeg_cmd,
                    path=path,
                    rate=rate,
                    output_filename=save_path,
                    offset=seg_start,
                    duration=max_duration,
                    fmin=f_min,
                    fmax=f_max,
                    denoise=denoise
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
                        idx + 1, f"{seg_start:.3f}", f"{seg_start + max_duration:.3f}",
                        f"{f_min:.1f}", f"{f_max:.1f}", output_name
                    ])

        raw_audio = [f.result() for f in futures]
        for i, audio_data in enumerate(raw_audio):
            results[i]["raw_audio"] = audio_data

    if download:
        f_csv.close()

    return results
                
                
# Segment ROI clips
rois = segment_rois("wat.mp3", rate=48000, csv_filename="rois.csv")
print(rois)