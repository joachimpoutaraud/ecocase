import os
import subprocess
import numpy as np

import cv2
import librosa

class BirdROI:
    def __init__(self, audio_path, sr=48000, fmin=20, fmax=15000, fps=25,
                 min_duration=0.5, max_duration=None,
                 audio_filter_chain=None, image_filter_chain=None, 
                 fscale="mel", n_fft=2048, hop_length=512):
        """
        Initializes the BirdROI with parameters for audio processing and ROI extraction.
        """
        
        self.audio_path = audio_path
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.fps = fps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.fscale = fscale
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.audio_filter_chain = audio_filter_chain or (
            f"aresample=resampler=soxr:out_sample_rate={self.sr},"
            f"firequalizer=gain='if(lt(f,{self.fmin}),-INF,if(gt(f,{self.fmax}),-INF,0))',"
            f"dynaudnorm=b=enabled"
        )

        self.image_filter_chain = image_filter_chain or (
            f"{self.audio_filter_chain},"
            f"showcwt=s=1x{self.n_fft}:slide=scroll:pps={self.hop_length}:"
            f"rate={self.fps}:deviation=10:min={self.fmin}:max={self.fmax},"
            f"fps={self.fps},format=yuv420p,extractplanes=y,"
            f"tblend=all_mode=hardmix,"
        )

        self.rois = self._extract_rois()

    def _get_duration(self):
        duration = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", self.audio_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.duration = float(duration.stdout.strip())
        return self.duration
    
    def _get_image(self, duration, output_path='cwt.png', preprocess=True):
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-v", "error",
            "-ss", "0", "-t", str(duration), "-i", self.audio_path,
            "-filter_complex", self.image_filter_chain + f"select=gte(n\\,1),tile={int(self.fps * duration)}x1", 
            "-frames:v", "1", "-pix_fmt", "gray", "-vcodec", "png", "-f", "image2pipe", "pipe:1"
        ]

        # Pipe image into memory and decode it
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        # Optionally, apply blur and morphological operations
        if preprocess:
            result = self.apply_blur_morpho(result.stdout, close_iters=2, dilate_iters=2)
        # Decode the image
        image = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_GRAYSCALE)

        # Remap y-axis to mel or log if requested
        if self.fscale in ("log", "mel"):
            height = image.shape[0]
            new_image = np.zeros_like(image)
            self._y_to_freq = np.zeros(height)
            for y in range(height):
                if self.fscale == "log":
                    log_f_min, log_f_max = np.log10(self.fmin), np.log10(self.fmax)
                    frac = y / (height - 1)
                    log_f = log_f_max - frac * (log_f_max - log_f_min)
                    freq = 10 ** log_f
                    y_lin = int((self.fmax - freq) / (self.fmax - self.fmin) * (height - 1))
                elif self.fscale == "mel": 
                    mel_f_min = librosa.hz_to_mel(self.fmin)
                    mel_f_max = librosa.hz_to_mel(self.fmax)
                    frac = y / (height - 1)
                    mel = mel_f_max - frac * (mel_f_max - mel_f_min)
                    freq = librosa.mel_to_hz(mel)
                    y_lin = int((self.fmax - freq) / (self.fmax - self.fmin) * (height - 1))
                y_lin = np.clip(y_lin, 0, height - 1)
                new_image[y, :] = image[y_lin, :]
                self._y_to_freq[y] = freq
            image = new_image
        else:
            # For linear, just store the mapping
            height = image.shape[0]
            self._y_to_freq = np.array([self.fmax - ((y / (height - 1)) * (self.fmax - self.fmin)) for y in range(height)])

        if output_path:
            cv2.imwrite(output_path, image)

        self.image = image
        return self.image


    def apply_blur_morpho(self, image_bytes, close_iters=2, dilate_iters=2, kernel=(16,16)):
        total_iters = close_iters + dilate_iters

        # Build filter_complex
        filter_lines = []
        # Generate one ellipse and split it
        filter_lines.append(
            f"color=black:s={kernel[0]}x{kernel[1]},format=gray8"
            ",geq=lum='255*lte(hypot((((2*X+1)/W-1)*cos(PI/4)-((2*Y+1)/H-1)*sin(PI/4))/0.5,"
            "(((2*X+1)/W-1)*sin(PI/4)+((2*Y+1)/H-1)*cos(PI/4))/1),1)',format=gray[ellipse]"
        )
        filter_lines.append(f"[ellipse]split={total_iters}" + "".join([f"[el{i+1}]" for i in range(total_iters)]))
        # Blur input
        filter_lines.append("[0:v]gblur=sigma=0.5[blurred]")

        # Apply morpho filters
        prev = "blurred"
        for i in range(close_iters):
            filter_lines.append(f"[{prev}][el{i+1}]morpho=mode=close[d{i+1}]")
            prev = f"d{i+1}"
        for i in range(dilate_iters):
            idx = close_iters + i
            filter_lines.append(f"[{prev}][el{idx+1}]morpho=mode=dilate[d{idx+1}]")
            prev = f"d{idx+1}"

        filter_lines.append(f"[{prev}]copy[out]")
        filter_complex = ";".join(filter_lines)

        # FFmpeg command
        morpho_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-v", "error",
            "-f", "image2pipe", "-vcodec", "png", "-i", "pipe:0",
            "-filter_complex", filter_complex,
            "-map", "[out]", "-frames:v", "1", "-pix_fmt", "gray", "-vcodec", "png",
            "-f", "image2pipe", "pipe:1"
        ]

        # Run FFmpeg
        p = subprocess.Popen(morpho_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result, err = p.communicate(input=image_bytes)

        if p.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {err.decode()}")

        return result

    def _find_rois(self, image, min_time_px=20, min_freq_px=20):
        # Preprocess image: blur and morphological operations to enhance blobs
        # # Gaussian blur to smooth noise
        # blurred = cv2.GaussianBlur(image, (5, 5), 0)
        # # Adaptive threshold or Otsu's threshold to binarize
        # _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # # Morphological closing to fill gaps and make blobs chunkier
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        # morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        # # Dilate to further thicken blobs
        # image = cv2.dilate(morphed, kernel, iterations=2)
        
        # Find contours in the processed image
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []
        for cnt in contours:
            if cnt is not None and len(cnt) > 0:
                x, y, w, h = cv2.boundingRect(cnt.astype(np.int32))
            if w > min_time_px and h > min_freq_px:
                rois.append({"bbox": (x, y, w, h)})

        return rois

    def _extract_rois(self):
        try:
            duration = self._get_duration()
            image = self._get_image(duration)
            rois_bounds = self._find_rois(image)

            # Get the height of the image
            height = image.shape[0]
            rois = []
            # Extract ROIs based on bounding boxes
            for roi in rois_bounds:
                x, y, w, h = roi["bbox"]
                # Convert width in pixels to duration in seconds
                duration = w / self.fps
                if duration < self.min_duration or (self.max_duration is not None and duration > self.max_duration):
                    continue  # Skip ROIs that do not meet duration criteria
                start, end = x / self.fps, (x + w) / self.fps
                roi_fmin = self._y_to_freq[min(y + h, height - 1)]
                roi_fmax = self._y_to_freq[max(y, 0)]
                rois.append({
                    "file": self.audio_path,
                    "fmin": roi_fmin,
                    "fmax": roi_fmax,
                    "start": start,
                    "end": end
                })
                
            return rois
            
        except Exception as e:
            print(f"❌ Failed to extract ROIs: {e}")
            return []
        
    def download_rois(self, output_dir="rois"):
        os.makedirs(output_dir, exist_ok=True)
        for idx, roi in enumerate(self.rois):
            fmin = roi["fmin"]
            fmax = roi["fmax"]
            start = roi["start"]
            end = roi["end"]
            duration = end - start
            base = os.path.splitext(os.path.basename(self.audio_path))[0]
            filename = f"{base}_roi_{idx+1}_f_{int(fmin)}-{int(fmax)}_t_{int(start)}-{int(end)}.wav"
            output_path = os.path.join(output_dir, filename)

            audio_filter_chain = (
                f"aresample=resampler=soxr:out_sample_rate={self.sr},"
                f"firequalizer=gain='if(lt(f,{fmin}),-INF,if(gt(f,{fmax}),-INF,0))',"
                f"dynaudnorm=b=enabled"
            )

            ffmpeg_cmd = [
                "ffmpeg", "-y", "-hide_banner", "-v", "error",
                "-ss", str(start), "-t", str(duration), "-i", self.audio_path,
                "-af", audio_filter_chain,
                "-ac", "1", "-ar", str(self.sr),
                output_path
            ]

            try:
                subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Failed to export ROI #{idx+1}: {e}")
                
    def plot_rois(self, figsize=(10, 6), box_color="red", box_linewidth=2):
        import matplotlib.pyplot as plt
        import librosa.display

        # Load audio
        y, sr = librosa.load(self.audio_path, sr=self.sr)
        # Compute spectrum
        if self.fscale == 'mel':
            S = librosa.feature.melspectrogram(y=y, n_fft=self.n_fft, hop_length=self.hop_length, fmax=self.fmax, fmin=self.fmin, sr=sr)
        else:
            S = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

        _, ax = plt.subplots(figsize=figsize)
        img = librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis=self.fscale,
            fmin=self.fmin,
            fmax=self.fmax,
            ax=ax
        )
        plt.colorbar(img, ax=ax, format="%+2.0f dB")

        # Draw bounding boxes for ROIs
        for roi in self.rois:
            ax.add_patch(
            plt.Rectangle(
                (roi["start"], min(roi["fmin"], roi["fmax"])),
                roi["end"] - roi["start"],
                abs(roi["fmax"] - roi["fmin"]),
                edgecolor=box_color,
                facecolor='none',
                linewidth=box_linewidth
            )
            )

        ax.set_title("Spectrogram with ROI Bounding Boxes")
        plt.tight_layout()
        plt.show()

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
                

if __name__ == "__main__":
    
    bird = BirdROI(audio_path="Turdus_iliacus/XC101610.wav", sr=48000, fmin=150, fmax=8000, min_duration=0.5, fscale="mel")
    bird.plot_rois()
    # bird.download_rois()

    # "Tinamus major/64136_24.00-27.00.wav"
    # "data/SMA04786_20220113_182200.wav"
    # "Nyctibius bracteatus/5071_3.00-6.00.wav"






# import subprocess
# import multiprocessing
# import os
# import cv2
# import numpy as np
# import pandas as pd
# from tqdm import tqdm


# class FFbird:
#     def __init__(self, sr=48000, fmin=20, fmax=15000, fps=25):
#         self.sr = sr
#         self.fmin = fmin
#         self.fmax = fmax
#         self.fps = fps
#         self.df = None

#     def build_ffmpeg_command(
#         self, audio_path, start, duration, fmin, fmax,
#         output_path=None, mode='audio', output_mode='disk',
#         audio_filter=None, image_filter=None
#     ):
#         base_cmd = ["ffmpeg", "-y", "-hide_banner", "-v", "error"]
#         segment = ["-ss", str(start), "-t", str(duration), "-i", audio_path]

#         # Use custom filter if provided, otherwise fall back to defaults
#         audio_filter_str = audio_filter or (
#             f"aresample=resampler=soxr:out_sample_rate={self.sr},"
#             f"firequalizer=gain='if(lt(f,{fmin}),-INF,if(gt(f,{fmax}),-INF,0))',"
#             f"dynaudnorm=b=enabled"
#         )
        
#         image_filter_str = image_filter or (
#             f"{audio_filter_str},"
#             f"showcwt=s=1x2048:iscale=log:scale=log:slide=scroll:"
#             f"rate={self.fps}:deviation=10:min={fmin}:max={fmax},"
#             f"fps={self.fps},format=yuv420p,extractplanes=y,"
#             f"dilation=threshold0=10,dilation=threshold0=10,dilation=threshold0=10,dilation=threshold0=10,dilation=threshold0=10,dilation=threshold0=10,"
#             f"tblend=all_mode=hardmix,"
#             f"select=gte(n\\,1),tile={int(self.fps * duration)}x1"
#         )

#         if mode == 'audio':
#             audio_cmd = ["-af", audio_filter_str, "-ac", "1", "-ar", str(self.sr)]
#             return base_cmd + segment + audio_cmd + (
#                 [output_path] if output_mode == "disk" else ["-f", "wav", "pipe:1"]
#             )

#         elif mode == 'image':
#             return base_cmd + segment + ["-filter_complex", image_filter_str] + (
#                 ["-frames:v", "1", output_path] if output_mode == "disk" else
#                 ["-frames:v", "1", "-f", "image2pipe", "-pix_fmt", "gray", "-vcodec", "png", "pipe:1"]
#             )

#         else:
#             raise ValueError("Invalid mode or output_mode")
        
#     def get_audio_duration(self, audio_path):
#         cmd = [
#             "ffprobe", "-v", "error",
#             "-show_entries", "format=duration",
#             "-of", "default=noprint_wrappers=1:nokey=1", audio_path
#         ]
#         result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         return float(result.stdout.strip())

#     def convert_pix_to_logfreq(self, y, height):
#         log_f_min, log_f_max = np.log10(self.fmin), np.log10(self.fmax)
#         log_f = log_f_max - (y / height) * (log_f_max - log_f_min)
#         return 10 ** log_f

#     def find_rois(self, image_array, min_duration, max_duration, min_time_px=10, min_freq_px=10):
#         height = image_array.shape[0]
#         contours, _ = cv2.findContours(image_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         rois_bounds = []

#         for cnt in contours:
#             if cnt is not None and len(cnt) > 0:
#                 x, y, w, h = cv2.boundingRect(cnt.astype(np.int32))
#                 if w >= min_time_px and h >= min_freq_px:
#                     merged = False
#                     for r in rois_bounds:
#                         rx, ry, rw, rh = r["bbox"]
#                         time_close = abs(x - (rx + rw)) <= min_time_px or abs(rx - (x + w)) <= min_time_px
#                         freq_overlap = not (y + h < ry - min_freq_px or ry + rh < y - min_freq_px)
#                         if time_close and freq_overlap:
#                             new_x, new_y = min(x, rx), min(y, ry)
#                             new_w = max(x + w, rx + rw) - new_x
#                             new_h = max(y + h, ry + rh) - new_y
#                             r["bbox"] = (new_x, new_y, new_w, new_h)
#                             merged = True
#                             break
#                     if not merged:
#                         rois_bounds.append({"bbox": (x, y, w, h)})

#         rois = []
#         for roi in rois_bounds:
#             x, y, w, h = roi["bbox"]
#             duration = w / self.fps
#             if duration >= min_duration and (max_duration is None or duration <= max_duration):
#                 start, end = x / self.fps, (x + w) / self.fps
#                 roi_fmin = self.convert_pix_to_logfreq(y + h, height)
#                 roi_fmax = self.convert_pix_to_logfreq(y, height)
#                 rois.append((roi_fmin, roi_fmax, start, end))

#         return rois

#     def _process_roi(self, args):
#         audio_path, min_duration, max_duration = args
#         try:

#             duration = self.get_audio_duration(audio_path)
#             ffmpeg_command = self.build_ffmpeg_command(
#                 audio_path, start=0, duration=duration,
#                 fmin=self.fmin, fmax=self.fmax, mode="image", output_mode="pipe"
#             )
#             result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, check=True)
#             image = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

#             rois = self.find_rois(image, min_duration, max_duration)
#             return [{
#                 "file": audio_path, "fmin": fmin, "fmax": fmax, "start": start, "end": end
#             } for fmin, fmax, start, end in rois]
#         except Exception as e:
#             print(f"⚠️ Error processing {audio_path}: {e}")
#             return []

#     def multiprocess_rois(self, path, min_duration=0.5, max_duration=None, csv_filename=None):
#         if os.path.isfile(path) and path.lower().endswith((".wav", ".mp3", ".flac")):
#             audio_files = [path]
#             print(f"🔍 Processing single audio file: '{path}'")
#         elif os.path.isdir(path):
#             audio_files = [
#                 os.path.join(path, f) for f in os.listdir(path)
#                 if os.path.isfile(os.path.join(path, f)) and f.lower().endswith((".wav", ".mp3", ".flac"))
#             ]
#             print(f"🔍 Found {len(audio_files)} audio files in folder '{path}'")
#         else:
#             raise ValueError("Provided path must be an audio file or directory.")

#         args = [(audio_path, min_duration, max_duration) for audio_path in audio_files]

#         results = []
#         with multiprocessing.Pool() as pool:
#             for result in tqdm(pool.imap_unordered(self._process_roi, args), total=len(args), desc="Processing audio files"):
#                 results.extend(result)

#         self.df = pd.DataFrame(results)
#         print(f"✅ ROI extraction complete: {len(self.df)} entries")

#         if csv_filename is not None:
#             self.df.to_csv(csv_filename, sep=";", encoding='utf-8', index=False, index_label='id')

#         return self.df

#     def _download_roi(self, args, image_filter=None):
#         audio_path, roi, index, output_dir, sr, mode = args
#         fmin, fmax, start, end = roi
#         duration = end - start
#         base = os.path.splitext(os.path.basename(audio_path))[0]

#         if mode in ["audio", "both"]:
#             audio_filename = f"{base}_roi_{index+1}_f_{int(fmin)}-{int(fmax)}_t_{int(start)}-{int(end)}.wav"
#             audio_path_out = os.path.join(output_dir, audio_filename)
#             ffmpeg_command = self.build_ffmpeg_command(
#                 audio_path, start, duration, fmin, fmax,
#                 output_path=audio_path_out, mode='audio', output_mode='disk'
#             )
#             subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#         if mode in ["image", "both"]:
#             image_filename = f"{base}_roi_{index+1}_f_{int(fmin)}-{int(fmax)}_t_{int(start)}-{int(end)}.png"
#             image_path_out = os.path.join(output_dir, image_filename)

#             # Simplified image filter to output filtered and normalized spectrum image
#             cwt_filter = image_filter or (
#                 f"aresample=resampler=soxr:out_sample_rate={sr},"
#                 f"firequalizer=gain='if(lt(f,{fmin}),-INF,if(gt(f,{fmax}),-INF,0))',"
#                 f"dynaudnorm=b=enabled,"
#                 f"showspectrumpic=fscale=log"
#             )

#             ffmpeg_command = self.build_ffmpeg_command(
#                 audio_path, start, duration, fmin, fmax,
#                 output_path=image_path_out,
#                 mode='image', output_mode='disk',
#                 image_filter=cwt_filter
#             )
#             subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#     def download_rois(self, df=None, output_folder="rois", mode="audio"):
#         os.makedirs(output_folder, exist_ok=True)

#         if df is not None:
#             if isinstance(df, str) and df.lower().endswith(".csv"):
#                 self.df = pd.read_csv(df)
#             elif isinstance(df, pd.DataFrame):
#                 self.df = df
#             else:
#                 raise ValueError("df must be a DataFrame or path to a CSV file.")

#         tasks = []
#         for i, row in self.df.iterrows():
#             audio_path = row["file"]
#             roi = (row["fmin"], row["fmax"], row["start"], row["end"])
#             tasks.append((audio_path, roi, i, output_folder, self.sr, mode))

#         label = {"audio": "Audio", "image": "Scaleogram", "both": "Audio + Image"}[mode]
#         print(f"🎧 Starting download of {len(tasks)} ROI(s): {label} → '{output_folder}'")

#         with multiprocessing.Pool() as pool:
#             list(tqdm(pool.imap_unordered(self._download_roi, tasks), total=len(tasks), desc=f"Export {label}"))

#         return self.df


# if __name__ == "__main__":
    
#     ffbird = FFbird(sr=48000, fmin=150, fmax=8000)
#     ffbird.multiprocess_rois(path="Tinamus major/64136_24.00-27.00.wav", min_duration=1.0, csv_filename='rois.csv')
#     ffbird.download_rois(output_folder="rois", mode="both")
    
    # "Tinamus major/64136_24.00-27.00.wav"
    # "SMA04786_20220113_182200.wav"
    # "Nyctibius bracteatus/5071_3.00-6.00.wav"


#################################
###### GET FREQUENCY RANGE ######
#################################

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
# folder_path = 'Crypturellus variegatus_Variegated Tinamou'

# fmin_list = []
# fmax_list = []

# for filename in os.listdir(folder_path):
#     if filename.endswith('.wav'):
#         file_path = os.path.join(folder_path, filename)

#         try:
#             y, sr = load_audio_ffmpeg(file_path)

#             # Use librosa's pyin to extract pitch
#             f0, voiced_flag, voiced_probs = librosa.pyin(
#                 y, fmin=150, fmax=8000, sr=sr,
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
#     mean_fmin = np.mean(fmin_list)
#     mean_fmax = np.mean(fmax_list)
#     print(f"\nMean fmin: {mean_fmin:.2f} Hz")
#     print(f"Mean fmax: {mean_fmax:.2f} Hz")
# else:
#     print("\nNo pitch data available to compute mean values.")