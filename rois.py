import os, subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2

from utils import convert_audio_to_image, convert_pixel_to_frequency


def extract_roi_audio(args):
    """
    Extracts and filters region of interests (ROIs) from a recording using FFmpeg.

    Parameters:
        path (str): Path to the input audio file.
        tmin (float): Start time of the ROI in seconds.
        tmax (float): End time of the ROI in seconds.
        fmin (float): Minimum frequency of the ROI in Hz.
        fmax (float): Maximum frequency of the ROI in Hz.
        audio_filter_chain (str): FFmpeg-compatible audio filter chain string.
        noise_reduction (int): Gain reduction outside the frequency band.
        download (bool): If True, saves filtered audio to disk and returns output path.
        output_folder (str): Folder to save the output audio file.
        encode (str): If 'audio', returns base64-encoded audio. If 'spectrogram', returns base64 image.
        rate (int): Sample rate for encoding and spectrogram.

    Returns:
        np.ndarray or str: Filtered audio data, base64 string, or output path.
    """
    
    path, tmin, tmax, fmin, fmax, audio_filter_chain, noise_reduction, download, rate = args

    full_filter_chain = f"firequalizer=gain='if(between(f,{int(fmin)},{int(fmax)}),0,{noise_reduction})'"

    if audio_filter_chain is not None:
        full_filter_chain = f"{audio_filter_chain},{full_filter_chain}"

    if download:
        full_filter_chain += ",asplit=2[a1][a2]"

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-v", "error",
        "-ss", str(tmin), "-t", str(tmax - tmin), "-i", path,
        "-filter_complex", full_filter_chain
    ]

    output_path = None

    if download:
        filename = os.path.splitext(os.path.basename(path))[0]
        output_folder = "rois"
        os.makedirs(output_folder, exist_ok=True)
        filename_roi = f"{filename}_t_{round(tmin,2)}_{round(tmax,2)}_f_{int(fmin)}_{int(fmax)}.wav"
        output_path = os.path.join(output_folder, filename_roi)
        cmd += ["-map", "[a1]", "-f", "f32le", "pipe:1", "-map", "[a2]", "-f", "wav", output_path]
    else:
        cmd += ["-f", "f32le", "pipe:1"]

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        audio_bytes, error = process.communicate()
        if error:
            print("FFmpeg error:", error.decode())

    roi_audio = np.frombuffer(audio_bytes, dtype=np.float32)

    if download:
        return roi_audio, output_path

    out = {
            "path": path,
            "rate": rate,
            "roi": roi_audio,
            "tmin": tmin,
            "tmax": tmax,
            "fmin": fmin,
            "fmax": fmax,
            "audio_filter_chain": audio_filter_chain
            }

    return out


def extract_roi_contours(args):
    audio_path, fmin, fmax, audio_filter_chain, noise_profile, n_fft, hop_length, fps, min_duration, max_duration, fscale, display = args

    try:
        image, height, width = convert_audio_to_image(audio_path, fmin, fmax, audio_filter_chain, noise_profile, n_fft, hop_length, fps)
        converted_image, px_to_freq = convert_pixel_to_frequency(image, fmin, fmax, height, fscale)

        filtered = cv2.bilateralFilter(converted_image, d=9, sigmaColor=75, sigmaSpace=75)
        blurred = cv2.GaussianBlur(filtered, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        border_thickness = 5
        border_mask = np.ones_like(thresh, dtype=np.uint8) * 255
        border_mask[border_thickness:-border_thickness, border_thickness:-border_thickness] = 0
        processed_image = cv2.bitwise_and(thresh, cv2.bitwise_not(border_mask))

        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            if cnt is None or len(cnt) == 0:
                continue

            x, y, w, h = cv2.boundingRect(cnt.astype(np.int32))
            duration_roi = w / fps

            if duration_roi < min_duration or (max_duration is not None and duration_roi > max_duration):
                continue
            if w >= width * 0.98 and h >= height * 0.98:
                continue

            boxes.append((x, y, x + w, y + h))

        rois = []
        for x1, y1, x2, y2 in boxes:
            tmin = x1 / fps
            tmax = x2 / fps
            roi_fmin = px_to_freq[min(int(y2), height - 1)]
            roi_fmax = px_to_freq[max(int(y1), 0)]

            rois.append({
                "path": audio_path,
                "tmin": tmin,
                "tmax": tmax,
                "fmin": roi_fmin,
                "fmax": roi_fmax,
                "min_duration": min_duration,
                "bounds": (x1, y1, x2, y2)
            })

        if display:
            fig, ax = plt.subplots()
            ax.imshow(converted_image, cmap="gray")
            for x1, y1, x2, y2 in boxes:
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                        linewidth=2, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
            ax.set_title(f"Detected ROIs in {os.path.basename(audio_path)}")
            plt.show()

        return rois
    except:
        pass