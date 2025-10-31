import subprocess
import numpy as np

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


def convert_audio_to_image(path, fmin=1, fmax=15000, audio_filter_chain=None, noise_profile=[0.0, 0.5], n_fft=512, hop_length=128, fps=50):

    duration = get_duration(path)
    noise_start, noise_end = noise_profile[0], noise_profile[1]
    noise_profile_duration = noise_end - noise_start

    filter_complex = (
        (f"[0:a]{audio_filter_chain},dynaudnorm=b=enabled[af];"
         if audio_filter_chain
         else "[0:a]dynaudnorm=b=enabled[af];"
        ) +
        
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