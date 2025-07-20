import subprocess
import multiprocessing
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

class FFbird:
    def __init__(self, sr=48000, fmin=20, fmax=15000, fps=25):
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.fps = fps
        self.df = None

    def get_audio_duration(self, audio_path):
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())

    def build_ffmpeg_command(
        self, audio_path, start, duration, fmin, fmax,
        output_path=None, mode='audio', output_mode='disk',
        audio_filter=None, image_filter=None
    ):
        base_cmd = ["ffmpeg", "-y", "-hide_banner", "-v", "error"]
        segment = ["-ss", str(start), "-t", str(duration), "-i", audio_path]

        # Use custom filter if provided, otherwise fall back to defaults
        audio_filter_str = audio_filter or (
            f"aresample=resampler=soxr:out_sample_rate={self.sr},"
            f"firequalizer=gain='if(lt(f,{fmin}),-INF,if(gt(f,{fmax}),-INF,0))',"
            f"dynaudnorm=b=enabled"
        )
        
        image_filter_str = image_filter or (
            f"{audio_filter_str},"
            f"showcwt=s=1x2048:iscale=log:scale=log:slide=scroll:"
            f"rate={self.fps}:deviation=10:min={fmin}:max={fmax},"
            f"fps={self.fps},format=yuv420p,extractplanes=y,"
            f"dilation=threshold0=10,dilation=threshold0=10,dilation=threshold0=10,dilation=threshold0=10,dilation=threshold0=10,dilation=threshold0=10,"
            f"tblend=all_mode=hardmix,"
            f"select=gte(n\\,1),tile={int(self.fps * duration)}x1"
        )

        if mode == 'audio':
            audio_cmd = ["-af", audio_filter_str, "-ac", "1", "-ar", str(self.sr)]
            return base_cmd + segment + audio_cmd + (
                [output_path] if output_mode == "disk" else ["-f", "wav", "pipe:1"]
            )

        elif mode == 'image':
            return base_cmd + segment + ["-filter_complex", image_filter_str] + (
                ["-frames:v", "1", output_path] if output_mode == "disk" else
                ["-frames:v", "1", "-f", "image2pipe", "-pix_fmt", "gray", "-vcodec", "png", "pipe:1"]
            )

        else:
            raise ValueError("Invalid mode or output_mode")

    def get_scaleogram(self, audio_path, fmin, fmax):
        duration = self.get_audio_duration(audio_path)
        cmd = self.build_ffmpeg_command(
            audio_path, start=0, duration=duration,
            fmin=fmin, fmax=fmax, mode="image", output_mode="pipe"
        )
        result = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
        return cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    def convert_pix_to_logfreq(self, y, height):
        log_f_min, log_f_max = np.log10(self.fmin), np.log10(self.fmax)
        log_f = log_f_max - (y / height) * (log_f_max - log_f_min)
        return 10 ** log_f

    def find_rois(self, image_array, min_duration, max_duration, min_time_px=10, min_freq_px=10):
        height = image_array.shape[0]
        contours, _ = cv2.findContours(image_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois_bounds = []

        for cnt in contours:
            if cnt is not None and len(cnt) > 0:
                x, y, w, h = cv2.boundingRect(cnt.astype(np.int32))
                if w >= min_time_px and h >= min_freq_px:
                    merged = False
                    for r in rois_bounds:
                        rx, ry, rw, rh = r["bbox"]
                        time_close = abs(x - (rx + rw)) <= min_time_px or abs(rx - (x + w)) <= min_time_px
                        freq_overlap = not (y + h < ry - min_freq_px or ry + rh < y - min_freq_px)
                        if time_close and freq_overlap:
                            new_x, new_y = min(x, rx), min(y, ry)
                            new_w = max(x + w, rx + rw) - new_x
                            new_h = max(y + h, ry + rh) - new_y
                            r["bbox"] = (new_x, new_y, new_w, new_h)
                            merged = True
                            break
                    if not merged:
                        rois_bounds.append({"bbox": (x, y, w, h)})

        rois = []
        for roi in rois_bounds:
            x, y, w, h = roi["bbox"]
            duration = w / self.fps
            if min_duration <= duration <= max_duration:
                start, end = x / self.fps, (x + w) / self.fps
                roi_fmin = self.convert_pix_to_logfreq(y + h, height)
                roi_fmax = self.convert_pix_to_logfreq(y, height)
                rois.append((roi_fmin, roi_fmax, start, end))
        return rois

    def _process_single_file(self, args):
        audio_path, min_duration, max_duration = args
        try:
            image = self.get_scaleogram(audio_path, fmin=self.fmin, fmax=self.fmax)
            rois = self.find_rois(image, min_duration, max_duration)
            return [{
                "file": audio_path, "fmin": fmin, "fmax": fmax, "start": start, "end": end
            } for fmin, fmax, start, end in rois]
        except Exception as e:
            print(f"⚠️ Error processing {audio_path}: {e}")
            return []

    def multiprocess_rois(self, path, min_duration=0.5, max_duration=10.0, csv_filename=None):
        if os.path.isfile(path) and path.lower().endswith((".wav", ".mp3", ".flac")):
            audio_files = [path]
            print(f"🔍 Processing single audio file: '{path}'")
        elif os.path.isdir(path):
            audio_files = [
                os.path.join(path, f) for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f)) and f.lower().endswith((".wav", ".mp3", ".flac"))
            ]
            print(f"🔍 Found {len(audio_files)} audio files in folder '{path}'")
        else:
            raise ValueError("Provided path must be an audio file or directory.")

        args = [(audio_path, min_duration, max_duration) for audio_path in audio_files]

        results = []
        with multiprocessing.Pool() as pool:
            for result in tqdm(pool.imap_unordered(self._process_single_file, args), total=len(args), desc="Processing audio files"):
                results.extend(result)

        self.df = pd.DataFrame(results)
        print(f"✅ ROI extraction complete: {len(self.df)} entries")

        if csv_filename is not None:
            self.df.to_csv(csv_filename, sep=";", encoding='utf-8', index=False, index_label='id')

        return self.df

    def _download_roi(self, args):
        audio_path, roi, index, output_dir, sr, mode = args
        fmin, fmax, start, end = roi
        duration = end - start
        base = os.path.splitext(os.path.basename(audio_path))[0]

        if mode in ["audio", "both"]:
            audio_filename = f"{base}_roi_{index+1}_f_{int(fmin)}-{int(fmax)}_t_{int(start)}-{int(end)}.wav"
            audio_path_out = os.path.join(output_dir, audio_filename)
            audio_cmd = self.build_ffmpeg_command(
                audio_path, start, duration, fmin, fmax,
                output_path=audio_path_out, mode='audio', output_mode='disk'
            )
            subprocess.run(audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if mode in ["image", "both"]:
            image_filename = f"{base}_roi_{index+1}_f_{int(fmin)}-{int(fmax)}_t_{int(start)}-{int(end)}.png"
            image_path_out = os.path.join(output_dir, image_filename)

            # Simplified image filter (just showcwt with default vertical resolution, no tile/postprocessing)
            cwt_filter = (
                f"aresample=resampler=soxr:out_sample_rate={sr},"
                f"firequalizer=gain='if(lt(f,{fmin}),-INF,if(gt(f,{fmax}),-INF,0))',"
                f"dynaudnorm=b=enabled,"
                f"showspectrumpic=fscale=log"
            )

            image_cmd = self.build_ffmpeg_command(
                audio_path, start, duration, fmin, fmax,
                output_path=image_path_out,
                mode='image', output_mode='disk',
                image_filter=cwt_filter
            )
            subprocess.run(image_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def download_rois(self, df=None, output_folder="rois", mode="audio"):
        os.makedirs(output_folder, exist_ok=True)

        if df is not None:
            if isinstance(df, str) and df.lower().endswith(".csv"):
                self.df = pd.read_csv(df)
            elif isinstance(df, pd.DataFrame):
                self.df = df
            else:
                raise ValueError("df must be a DataFrame or path to a CSV file.")

        tasks = []
        for i, row in self.df.iterrows():
            audio_path = row["file"]
            roi = (row["fmin"], row["fmax"], row["start"], row["end"])
            tasks.append((audio_path, roi, i, output_folder, self.sr, mode))

        label = {"audio": "Audio", "image": "Scaleogram", "both": "Audio + Image"}[mode]
        print(f"🎧 Starting download of {len(tasks)} ROI(s): {label} → '{output_folder}'")

        with multiprocessing.Pool() as pool:
            list(tqdm(pool.imap_unordered(self._download_roi, tasks), total=len(tasks), desc=f"Export {label}"))

        return self.df


if __name__ == "__main__":
    
    ffbird = FFbird(sr=48000, fmin=150, fmax=8000)
    ffbird.multiprocess_rois(path="Tinamus major/64136_24.00-27.00.wav", min_duration=1.0, max_duration=10.0, csv_filename='rois.csv')
    ffbird.download_rois(output_folder="rois", mode="both")
    
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