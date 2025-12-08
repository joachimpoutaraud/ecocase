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
        download (bool): If True, saves filtered audio to disk and returns output path.
        rate (int): Sample rate for encoding and spectrogram.

    Returns:
        If download is True:
            tuple: (roi_audio (np.ndarray), output_path (str))
        Else:
            dict: {"path", "rate", "roi", "tmin", "tmax", "fmin", "fmax", "audio_filter_chain"}
    """
    
    path, tmin, tmax, fmin, fmax, audio_filter_chain, download, rate = args

    # Denoise with FFT and apply FIR Equalization around ROI boundaries using arbitrary frequency response
    full_filter_chain = ",".join(["afftdn=tn=1", f"firequalizer=gain='if(between(f,{fmin},{fmax}),0,-INF)'"])    

    # Append user-defined filters if any
    if audio_filter_chain is not None:
        full_filter_chain = f"{audio_filter_chain},{full_filter_chain}"

    # Optional split for download
    if download:
        full_filter_chain += ",asplit=2[a1][a2]"

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-v", "error",
        "-ss", str(tmin), "-t", str(tmax - tmin), "-i", path,
        "-filter_complex", full_filter_chain
    ]

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
    audio_path, fmin, fmax, audio_filter_chain, noise_floor, noise_profile, n_fft, pps, min_duration, max_duration, fscale, display = args

    try:
        image, height, width = convert_audio_to_image(audio_path, fmin, fmax, audio_filter_chain, noise_floor, noise_profile, n_fft, pps)
        converted_image, px_to_freq = convert_pixel_to_frequency(image, fmin, fmax, height, fscale)

        kernel_dim, border_thickness = 5, 5
        # Replaces each pixel's value with a weighted average of its neighbors
        filtered = cv2.bilateralFilter(converted_image, d=kernel_dim, sigmaColor=75, sigmaSpace=75)
        # Apply Gaussian blur to reduce noise and improve thresholding
        blurred = cv2.GaussianBlur(filtered, (kernel_dim, kernel_dim), 0)
        # Perform automatic image thresholding with Otsu's method
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Remove border artifacts
        border_mask = np.ones_like(thresh, dtype=np.uint8) * 255
        border_mask[border_thickness:-border_thickness, border_thickness:-border_thickness] = 0
        processed_image = cv2.bitwise_and(thresh, cv2.bitwise_not(border_mask))
        # Find contours in the binary image
        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            if cnt is None or len(cnt) == 0:
                continue

            x, y, w, h = cv2.boundingRect(cnt.astype(np.int32))
            duration_roi = w / pps

            if duration_roi < min_duration or (max_duration is not None and duration_roi > max_duration):
                continue
            if w >= width * 0.98 and h >= height * 0.98:
                continue

            boxes.append((x, y, x + w, y + h))

        rois = []
        for x1, y1, x2, y2 in boxes:
            tmin = x1 / pps
            tmax = x2 / pps
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
            # Convert x-axis from pixels to time (seconds)
            num_ticks = 10
            tick_pos = np.linspace(0, width, num=num_ticks)
            tick_labels = [f"{(p / pps):.2f}" for p in tick_pos]
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel("Time (s)")
            ax.set_title(f"Detected ROIs in {os.path.basename(audio_path)}")
            plt.show()

        return rois
    except Exception:
        return []