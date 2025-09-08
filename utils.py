import os, subprocess
import io, base64
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.signal import spectrogram

import cv2
import librosa

def get_duration(path):
    duration = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration = float(duration.stdout.strip())
    return duration

def extract_roi(path, tmin, tmax, fmin, fmax, audio_filter_chain=None, noise_reduction=-20, download=False, output_folder="rois", encode=None, rate=48000):
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

    if encode == 'audio':
        buf = io.BytesIO()
        wavfile.write(buf, rate, roi_audio)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('ascii')
        return f"data:audio/wav;base64,{b64}"

    if encode == 'spectrogram':
        roi_audio_scaled = (roi_audio * 32767).astype(np.int16)
        f, t, Sxx = spectrogram(roi_audio_scaled, fs=rate)
        Sxx_db = 10 * np.log10(np.abs(Sxx)**2 + 1e-10)

        plt.figure(figsize=(4, 2))
        plt.pcolormesh(t, f, Sxx_db, shading='auto', cmap='gray')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title('Spectrogram')
        buf_img = io.BytesIO()
        plt.savefig(buf_img, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf_img.seek(0)
        img_b64 = base64.b64encode(buf_img.read()).decode('ascii')
        return f'<img src="data:image/png;base64,{img_b64}" />'

    return [roi_audio]

def convert_pixel_to_frequency(image, fmin, fmax, height, fscale="mel"):
    # Convert pixel to frequency scale
    converted_image = np.zeros_like(image)
    y_to_freq = np.zeros(height)

    for y in range(height):
        frac = y / (height - 1)
        if fscale == "log":
            log_f_min, log_f_max = np.log10(fmin), np.log10(fmax)
            log_f = log_f_max - frac * (log_f_max - log_f_min)
            freq = 10 ** log_f
        elif fscale == "mel":
            mel_f_min = librosa.hz_to_mel(fmin)
            mel_f_max = librosa.hz_to_mel(fmax)
            mel = mel_f_max - frac * (mel_f_max - mel_f_min)
            freq = librosa.mel_to_hz(mel)
        else:
            freq = fmax - frac * (fmax - fmin)

        y_lin = int((fmax - freq) / (fmax - fmin) * (height - 1))
        y_lin = np.clip(y_lin, 0, height - 1)
        converted_image[y, :] = image[y_lin, :]
        y_to_freq[y] = freq 

    return converted_image, y_to_freq

def convert_audio_to_image(path, fmin=1, fmax=15000, audio_filter_chain=None, noise_profile=[0.0, 0.5], n_fft=512, hop_length=128, fps=50):
    
    audio_filter_chain = audio_filter_chain or ",".join(["dynaudnorm=b=enabled"])

    duration = get_duration(path)
    noise_start, noise_end = noise_profile[0], noise_profile[1]
    noise_profile_duration = noise_end - noise_start

    filter_complex = (
        f"[0:a]{audio_filter_chain}[af];"
        
        # Split audio file in two
        f"[af]asplit=2[a1][a2];"
        
        # Trim noise profile from first audio split
        f"[a1]atrim=start={noise_start}:end={noise_end},asetpts=PTS-STARTPTS[a_noise];"
        f"[a_noise]aformat=sample_fmts=s16:channel_layouts=mono[a_formatted];"
        f"[a2]aformat=sample_fmts=s16:channel_layouts=mono[a_main];"

        # Concatenate noise profile and denoise second audio split
        f"[a_formatted][a_main]concat=n=2:v=0:a=1[a2_with_noise];"
        f"[a2_with_noise]asendcmd={noise_start} afftdn sn start,asendcmd={noise_end} afftdn sn end,afftdn=tn=1:tr=1[a2_denoised];"

        # Remove noise profile from second audio split
        f"[a2_denoised]atrim=start={noise_profile_duration},asetpts=PTS-STARTPTS[a_denoised];"
        
        # Convert audio to scaleogram image
        f"[a_denoised]showcwt=s=1x{n_fft}:slide=scroll:pps={hop_length}:rate={fps}:deviation=10:min={fmin}:max={fmax},fps={fps},tile={int(fps * duration)}x1[transform]"
    )

    # Create FFmpeg command
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-v", "error",
        "-ss", "0", "-t", str(duration), "-i", path,
        "-filter_complex", filter_complex,
        "-map", "[transform]", "-frames:v", "1", 
        "-pix_fmt", "gray", "-vcodec", "png",
        "-f", "image2pipe", "pipe:1"
    ]

    # Process FFmpeg command
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        image_bytes, error = process.communicate()
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        height, width = image.shape

    return image, height, width