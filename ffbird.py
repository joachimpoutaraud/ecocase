import subprocess
import numpy as np

import cv2
import librosa

from birdnetlib.main import RecordingBuffer
from birdnetlib.analyzer import Analyzer

class FFmpegROI():
    def __init__(self, audio_path, rate=48000, fmin=150, fmax=8000, fps=50,
                 audio_filter_chain=None, image_filter_chain=None, 
                 fscale="mel", n_fft=512, hop_length=128):
        self.audio_path = audio_path
        self.rate = rate
        self.fmin = fmin
        self.fmax = fmax
        self.fps = fps
        self.fscale = fscale
        self.n_fft = n_fft
        self.hop_length = hop_length

        # birdnetlib parameters
        self.analyzer = Analyzer()
        self.sample_secs = 3.0
        self.week_48 = -1
        self.date = None
        self.sensitivity = 1.0
        self.lat = None
        self.lon = None
        self.min_conf = 0.1
        self.overlap = 0.0
        self.return_all_detections = False

        self.audio_filter_chain = audio_filter_chain or (
            f"aresample=resampler=soxr:out_sample_rate={self.rate},"
            f"highpass=f={self.fmin},lowpass=f={self.fmax},"
            f"dynaudnorm=b=enabled"
        )

        self.image_filter_chain = image_filter_chain or (
            f"showcwt=s=1x{self.n_fft}:slide=scroll:pps={self.hop_length}:"
            f"rate={self.fps}:deviation=10:min={self.fmin}:max={self.fmax},"
            f"fps={self.fps}"
        )

        self.audio, self.image = self._load_audio_and_image()

    def _get_duration(self):
        duration = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            self.audio_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.duration = float(duration.stdout.strip())
        return self.duration

    def _load_audio_and_image(self, noise_profile=[0.0, 0.5], audio_output_path=None, image_output_path=None):
        self.duration = self._get_duration()
        noise_start = noise_profile[0]
        noise_end = noise_profile[1]
        profile_duration = noise_end - noise_start

        filter_complex = (
            f"[0:a]{self.audio_filter_chain}[pre];"
            f"[pre]asplit=2[a1][a2];"

            # Extract noise profile from audio
            f"[a1]atrim=start={noise_start}:end={noise_end},asetpts=PTS-STARTPTS[a_noise];"
            f"[a_noise]aformat=sample_fmts=s16:channel_layouts=mono[a_formatted];"
            f"[a2]aformat=sample_fmts=s16:channel_layouts=mono[a_main];"

            # Prepend noise profile and denoise
            f"[a_formatted][a_main]concat=n=2:v=0:a=1[a2_with_noise];"
            f"[a2_with_noise]asendcmd={noise_start} afftdn sn start,"
            f"asendcmd={noise_end} afftdn sn end,"
            f"afftdn=tn=1:tr=1[a2_denoised];"

            # Remove noise profile segment from audio
            f"[a2_denoised]atrim=start={profile_duration},asetpts=PTS-STARTPTS[denoised];"
            f"[denoised]asplit=2[audio][audio_img];"
            
            # Generate image from denoised audio
            f"[audio_img]{self.image_filter_chain},select=gte(n\,1),tile={int(self.fps * self.duration)}x1[image]"
        )

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-v", "error",
            "-ss", "0", "-t", str(self.duration), "-i", self.audio_path,
            "-filter_complex", filter_complex
        ]

        # Audio output
        cmd += ["-map", "[audio]"]
        if audio_output_path:
            cmd += ["-ac", "1", "-ar", str(self.rate), "-f", "s16le", audio_output_path]
            self.audio = librosa.load(audio_output_path, sr=self.rate)[0]
        else:
            cmd += ["-ac", "1", "-ar", str(self.rate), "-f", "s16le", "pipe:1"]

        # Image output
        cmd += ["-map", "[image]"]
        if image_output_path:
            cmd += ["-frames:v", "1", "-pix_fmt", "gray", "-vcodec", "png", "-f", "image2", image_output_path]
            self.image = cv2.imread(image_output_path, cv2.IMREAD_GRAYSCALE)
        else:
            cmd += ["-frames:v", "1", "-pix_fmt", "gray", "-vcodec", "png", "-f", "image2pipe", "pipe:2"]

            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
                audio_bytes, image_bytes = proc.communicate()

                self.audio = np.frombuffer(audio_bytes, dtype=np.int16)
                self.image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.imshow(self.image, cmap="gray")
        # plt.show()

        # Postprocess image for ROI analysis   
        filtered = cv2.bilateralFilter(self.image, d=9, sigmaColor=75, sigmaSpace=75)
        blurred = cv2.GaussianBlur(filtered, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Create a mask of the border area to remove artifacts
        border_thickness = 10
        border_mask = np.ones_like(thresh, dtype=np.uint8) * 255
        border_mask[border_thickness:-border_thickness, border_thickness:-border_thickness] = 0
        # Remove any foreground pixels near the edges
        self.image = cv2.bitwise_and(thresh, cv2.bitwise_not(border_mask))   

        return self.audio, self.image

    def _convert_pix_to_freq(self, image, height):
        # Frequency remapping
        converted_image = np.zeros_like(image)
        self._y_to_freq = np.zeros(height)

        for y in range(height):
            frac = y / (height - 1)
            if self.fscale == "log":
                log_f_min, log_f_max = np.log10(self.fmin), np.log10(self.fmax)
                log_f = log_f_max - frac * (log_f_max - log_f_min)
                freq = 10 ** log_f
            elif self.fscale == "mel":
                mel_f_min = librosa.hz_to_mel(self.fmin)
                mel_f_max = librosa.hz_to_mel(self.fmax)
                mel = mel_f_max - frac * (mel_f_max - mel_f_min)
                freq = librosa.mel_to_hz(mel)
            else:
                freq = self.fmax - frac * (self.fmax - self.fmin)

            y_lin = int((self.fmax - freq) / (self.fmax - self.fmin) * (height - 1))
            y_lin = np.clip(y_lin, 0, height - 1)
            converted_image[y, :] = image[y_lin, :]
            self._y_to_freq[y] = freq

        return converted_image

    def find_rois(self, min_duration=0.5, max_duration=None, margin_time=0.1, margin_freq=250, display=True):
        try:
            height, width = self.image.shape[:2]
            image = self._convert_pix_to_freq(self.image, height)
            
            # Find contours on the image
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for cnt in contours:
                if cnt is None or len(cnt) == 0:
                    continue

                x, y, w, h = cv2.boundingRect(cnt.astype(np.int32))
                duration_roi = w / self.fps
                
                if duration_roi < min_duration or (max_duration is not None and duration_roi > max_duration):
                    continue

                time_margin_px = int(margin_time * self.fps)
                freq_margin_px = int(margin_freq / (abs(self._y_to_freq[0] - self._y_to_freq[-1]) / height))    

                x_m = max(x - time_margin_px, 0)
                w_m = min(w + 2 * time_margin_px, width - x_m)
                y_m = max(y - freq_margin_px, 0)
                h_m = min(h + 2 * freq_margin_px, height - y_m)

                # Skip ROIs that span the entire image (both width and height)
                if w_m >= width * 0.98 and h_m >= height * 0.98:
                    continue

                boxes.append((x_m, y_m, x_m + w_m, y_m + h_m))

            self.rois = []
            plot_bounds = []

            for x1, y1, x2, y2 in boxes:
                start = x1 / self.fps
                end = x2 / self.fps
                roi_fmin = self._y_to_freq[min(int(y2), height - 1)]
                roi_fmax = self._y_to_freq[max(int(y1), 0)]

                self.rois.append({
                    "file": self.audio_path,
                    "fmin": roi_fmin,
                    "fmax": roi_fmax,
                    "start": start,
                    "end": end,
                })

                plot_bounds.append((x1, y1, x2 - x1, y2 - y1))

            if display:
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches

                fig, ax = plt.subplots()
                ax.imshow(image, cmap="gray")
                for x, y, w, h in plot_bounds:
                    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
                    ax.add_patch(rect)
                ax.set_title("Detected ROIs with Bounding Boxes")
                plt.show()

            return self.rois

        except Exception as e:
            print(f"❌ Failed to extract ROIs: {e}")
            return []

    # def plot_rois(self, figsize=(10, 6), box_color="red", box_linewidth=2):
    #     import matplotlib.pyplot as plt
    #     import librosa.display

    #     # Load audio
    #     y, sr = librosa.load(self.audio_path, sr=self.rate)
    #     # Compute spectrum
    #     if self.fscale == 'mel':
    #         S = librosa.feature.melspectrogram(y=y, n_fft=self.n_fft, n_mels=32, hop_length=self.hop_length, fmax=self.fmax, fmin=self.fmin, sr=sr)
    #     else:
    #         S = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
    #     S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    #     _, ax = plt.subplots(figsize=figsize)
    #     img = librosa.display.specshow(
    #         S_db,
    #         sr=sr,
    #         hop_length=self.hop_length,
    #         x_axis='time',
    #         y_axis=self.fscale,
    #         fmin=self.fmin,
    #         fmax=self.fmax,
    #         ax=ax
    #     )
    #     plt.colorbar(img, ax=ax, format="%+2.0f dB")

    #     # Draw bounding boxes for ROIs
    #     for roi in self.rois:
    #         ax.add_patch(
    #         plt.Rectangle(
    #             (roi["start"], min(roi["fmin"], roi["fmax"])),
    #             roi["end"] - roi["start"],
    #             abs(roi["fmax"] - roi["fmin"]),
    #             edgecolor=box_color,
    #             facecolor='none',
    #             linewidth=box_linewidth
    #         )
    #         )

    #     ax.set_title("Spectrogram with ROI Bounding Boxes")
    #     plt.tight_layout()
    #     plt.show()     


if __name__ == "__main__":
    # TODO For repertoire analysis noise profile need to be computed on the whole audio file not 3s segments, to be continued
    
    import time
    from pathlib import Path

    start_time = time.time()

    # # Set the folder path
    # folder_path = Path("Nyctibius bracteatus")
    # # Iterate over all .wav files in the folder
    # for audio_file in folder_path.glob("*.wav"):
    #     bird = BirdROI(audio_path=audio_file, fscale="mel", fmin=150, fmax=8000)
    #     bird.find_rois(min_duration=0.5, display=True)
        
    bird = FFmpegROI(audio_path="turdus.wav", fscale="mel", fmin=150, fmax=8000)
    bird.find_rois(min_duration=0.5, display=True)
                  
    # bird.extract_embeddings()
    # print(bird.embeddings)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    
    # bird.plot_rois()
    # bird.download_rois()

    # "Turdus_iliacus/XC101610.wav"
    # "SMA04635_20220318_182700.wav"
    
    # "Tinamus major/64136_24.00-27.00.wav"
    # "data/SMA04786_20220113_182200.wav"
    # "Nyctibius bracteatus/5071_3.00-6.00.wav"







    # def plot_rois(self, figsize=(6, 12), box_color=(255, 0, 0), box_thickness=2):
    #     import matplotlib.pyplot as plt
    #     img_rgb = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

    #     for roi in self.rois:
    #         # Convert start/end to pixel positions
    #         x = int(roi["start"] * self.fps)
    #         w = int((roi["end"] - roi["start"]) * self.fps)
            
    #         # Convert frequency bounds to vertical positions (reversed y-axis)
    #         height = self.image.shape[0]
    #         fmin = roi["fmin"]
    #         fmax = roi["fmax"]
            
    #         if self.fscale == "log":
    #             def freq_to_y(f):  # reverse of log mapping
    #                 log_f_min, log_f_max = np.log10(self.fmin), np.log10(self.fmax)
    #                 return int(height * (1 - (np.log10(f) - log_f_min) / (log_f_max - log_f_min)))
    #         elif self.fscale == "mel":
    #             def freq_to_y(f): # reverse of mel mapping
    #                 mel = librosa.hz_to_mel(f)
    #                 mel_min = librosa.hz_to_mel(self.fmin)
    #                 mel_max = librosa.hz_to_mel(self.fmax)
    #                 return int(height * (1 - (mel - mel_min) / (mel_max - mel_min)))
    #         else:
    #             def freq_to_y(f):  # reverse of linear mapping
    #                 return int(height * (1 - (f - self.fmin) / (self.fmax - self.fmin)))

    #         y1 = freq_to_y(fmax)
    #         y2 = freq_to_y(fmin)
    #         h = y2 - y1
            
    #         # Draw the bounding boxes
    #         cv2.rectangle(img_rgb, (x, y1), (x + w, y1 + h), box_color, box_thickness)

    #     plt.figure(figsize=figsize)
    #     plt.imshow(img_rgb)
    #     plt.title("ROIs with Bounding Boxes")
    #     plt.axis("off")
    #     plt.show()
    
    # def apply_blur_morpho(self, image_bytes, close_iters=2, dilate_iters=2, kernel=(7,7)):
    #     total_iters = close_iters + dilate_iters

    #     # Build filter_complex
    #     filter_lines = []
    #     # Generate one ellipse and split it
    #     filter_lines.append(
    #         f"color=black:s={kernel[0]}x{kernel[1]}"
    #         ",geq=lum='255*lte(hypot((((2*X+1)/W-1)*cos(PI/4)-((2*Y+1)/H-1)*sin(PI/4))/0.5,"
    #         "(((2*X+1)/W-1)*sin(PI/4)+((2*Y+1)/H-1)*cos(PI/4))/1),1)',format=gray[ellipse]"
    #     )
    #     filter_lines.append(f"[ellipse]split={total_iters}" + "".join([f"[el{i+1}]" for i in range(total_iters)]))

    #     # Blur input
    #     filter_lines.append("[0:v]smartblur[blurred]")
    #     # Apply morpho filters
    #     prev = "blurred"
    #     for i in range(close_iters):
    #         filter_lines.append(f"[{prev}][el{i+1}]morpho=mode=close[d{i+1}]")
    #         prev = f"d{i+1}"
    #     for i in range(dilate_iters):
    #         idx = close_iters + i
    #         filter_lines.append(f"[{prev}][el{idx+1}]morpho=mode=dilate[d{idx+1}]")
    #         prev = f"d{idx+1}"

    #     filter_lines.append(f"[{prev}]copy[out]")
    #     filter_complex = ";".join(filter_lines)

    #     # Pipe FFmpeg command
    #     morpho_cmd = [
    #         "ffmpeg", "-y", "-hide_banner", "-v", "error",
    #         "-f", "image2pipe", "-vcodec", "png", "-i", "pipe:0",
    #         "-filter_complex", filter_complex,
    #         "-map", "[out]", "-frames:v", "1", "-pix_fmt", "gray", "-vcodec", "png",
    #         "-f", "image2pipe", "pipe:1"
    #     ]

    #     p = subprocess.Popen(morpho_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     result, err = p.communicate(input=image_bytes)

    #     if p.returncode != 0:
    #         raise RuntimeError(f"FFmpeg failed: {err.decode()}")

    #     return result



# ################################
# ##### GET FREQUENCY RANGE ######
# ################################

# import subprocess
# import numpy as np
# import librosa
# import os

# def load_audio_ffmpeg(filepath, sr=22050):
#     cmd = [
#         'ffmpeg',
#         '-i', filepath,
#         '-f', 'f32le',
#         '-ac', '1',
#         '-ar', str(sr),
#         '-loglevel', 'quiet',
#         '-'
#     ]
#     process = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
#     audio = np.frombuffer(process.stdout, dtype=np.float32)
#     return audio, sr

# # Folder with audio chunks
# folder_path = 'Nyctibius bracteatus'

# fmin_list = []
# fmax_list = []

# for filename in os.listdir(folder_path):
#     if filename.endswith('.wav'):
#         file_path = os.path.join(folder_path, filename)

#         try:
#             y, sr = load_audio_ffmpeg(file_path)
#             # Use librosa's pyin to extract pitch
#             f0, voiced_flag, voiced_probs = librosa.pyin(
#                 y, fmin=25, fmax=10000, sr=sr,
#                 frame_length=1024, hop_length=512
#             )

#             # Filter out unvoiced regions
#             f0_clean = f0[~np.isnan(f0)]

#             if len(f0_clean) > 0:
#                 fmin = np.min(f0_clean)
#                 fmax = np.max(f0_clean)
#                 fmin_list.append(fmin)
#                 fmax_list.append(fmax)
#                 print(f"{filename}: fmin = {fmin:.2f} Hz, fmax = {fmax:.2f} Hz")
#             else:
#                 print(f"{filename}: No voiced pitch detected.")

#         except Exception as e:
#             print(f"Error processing {filename}: {e}")

# # 🔚 Calculate overall mean values
# if fmin_list and fmax_list:
#     mean_fmin = np.median(fmin_list)
#     mean_fmax = np.median(fmax_list)
#     print(f"\nMean fmin: {mean_fmin:.2f} Hz")
#     print(f"Mean fmax: {mean_fmax:.2f} Hz")
# else:
#     print("\nNo pitch data available to compute mean values.")