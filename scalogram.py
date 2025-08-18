import os, sys, contextlib
import subprocess
import numpy as np

import cv2
import librosa

from birdnetlib.main import RecordingBuffer
from birdnetlib.analyzer import Analyzer

# ——— Suppress stdout/stderr ——— #
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

class ImageROI():
    def __init__(self, path, rate=48000, fmin=150, fmax=8000, fps=50, 
                 fscale="mel", n_fft=512, hop_length=128):
        
        self.path = path
        self.rate = rate
        self.fmin = fmin
        self.fmax = fmax
        self.fps = fps
        self.fscale = fscale
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.image = self.convert_audio_to_image()

    def _get_duration(self):
        duration = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            self.path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.duration = float(duration.stdout.strip())
        return self.duration

    def _convert_px_to_fscale(self, image):
        # Frequency remapping
        converted_image = np.zeros_like(image)
        self._y_to_freq = np.zeros(self.height)

        for y in range(self.height):
            frac = y / (self.height - 1)
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

            y_lin = int((self.fmax - freq) / (self.fmax - self.fmin) * (self.height - 1))
            y_lin = np.clip(y_lin, 0, self.height - 1)
            converted_image[y, :] = image[y_lin, :]
            self._y_to_freq[y] = freq

        return converted_image

    def convert_audio_to_image(self, noise_profile=[0.0, 0.5], audio_filter_chain=None, image_filter_chain=None):
        
        self.audio_filter_chain = audio_filter_chain or ",".join([
            f"aresample=resampler=soxr:out_sample_rate={self.rate}",
            f"highpass=f={self.fmin}", f"lowpass=f={self.fmax}",
            f"dynaudnorm=b=enabled"
        ])

        self.image_filter_chain = image_filter_chain or ",".join([
            f"showcwt=s=1x{self.n_fft}:slide=scroll:pps={self.hop_length}:"
            f"rate={self.fps}:deviation=10:min={self.fmin}:max={self.fmax}"
            # f"showspectrum=s=1x{self.n_fft}:slide=scroll:"
            # f"start={self.fmin}:stop={self.fmax}"
        ])

        noise_start, noise_end = noise_profile[0], noise_profile[1]
        noise_profile_duration = noise_end - noise_start
        
        self.duration = self._get_duration()

        filter_complex = (
            f"[0:a]{self.audio_filter_chain}[af];"
            
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
            
            # Convert audio to image
            f"[a_denoised]{self.image_filter_chain},fps={self.fps},tile={int(self.fps * self.duration)}x1[transform]"
        )

        # Create FFmpeg command
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-v", "error",
            "-ss", "0", "-t", str(self.duration), "-i", self.path,
            "-filter_complex", filter_complex,
            "-map", "[transform]", "-frames:v", "1", 
            "-pix_fmt", "gray", "-vcodec", "png",
            "-f", "image2pipe", "pipe:1"
        ]

        # Process FFmpeg command
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            image_bytes, error = process.communicate()
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            self.height, self.width = image.shape
            
        # Convert pixel to frequency scale
        self.image = self._convert_px_to_fscale(image)  

        return self.image

    def detect_rois(self, min_duration=0.5, max_duration=None, margin_time=0.1, margin_freq=250, display=True):
        try:      
            # Postprocess gray image for ROI analysis   
            filtered = cv2.bilateralFilter(self.image, d=9, sigmaColor=75, sigmaSpace=75)
            blurred = cv2.GaussianBlur(filtered, (7, 7), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Create a mask to remove border artifacts
            border_thickness = 5
            border_mask = np.ones_like(thresh, dtype=np.uint8) * 255
            border_mask[border_thickness:-border_thickness, border_thickness:-border_thickness] = 0
            
            # Remove foreground pixels near the edges
            processed_image = cv2.bitwise_and(thresh, cv2.bitwise_not(border_mask)) 
                  
            # Find contours on the processed image
            contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for cnt in contours:
                if cnt is None or len(cnt) == 0:
                    continue

                x, y, w, h = cv2.boundingRect(cnt.astype(np.int32))
                duration_roi = w / self.fps
                
                # Skip ROIs that are shorter than min_duration and, optionally, longer than max_duration
                if duration_roi < min_duration or (max_duration is not None and duration_roi > max_duration):
                    continue
                  
                # Skip ROIs that span the entire image (both width and height)
                if w >= self.width * 0.98 and h >= self.height * 0.98:
                    continue
                
                boxes.append((x, y, x + w, y + h))

            self.rois = []

            for x1, y1, x2, y2 in boxes:
                tmin = x1 / self.fps
                tmax = x2 / self.fps
                fmin = self._y_to_freq[min(int(y2), self.height - 1)]
                fmax = self._y_to_freq[max(int(y1), 0)]

                self.rois.append({
                    "path": self.path,
                    "rate": self.rate,
                    "tmin": tmin,
                    "tmax": tmax,
                    "fmin": fmin,
                    "fmax": fmax,
                    "min_duration": min_duration,
                    "bounds": (x1, y1, x2, y2)  # Store raw box coordinates for display
                })

            if display:
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches

                fig, ax = plt.subplots()
                ax.imshow(self.image, cmap="gray")
                for roi in self.rois:
                    x1, y1, x2, y2 = roi["bounds"]
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                            linewidth=2, edgecolor='lime', facecolor='none')
                    ax.add_patch(rect)
                ax.set_title("Detected ROIs with Bounding Boxes")
                plt.show()

            return self.rois

        except Exception as e:
            print(f"❌ Failed to extract ROIs: {e}")
            return []
        
        
class AudioROI(RecordingBuffer):
    def __init__(self, path, tmin=0.0, tmax=None, fmin=150, fmax=8000, rate=48000, min_duration=0.5, audio_filter_chain=None, download=True):
        
        self.path = path
        self.tmin = tmin
        self.tmax = tmax
        self.fmin = fmin
        self.fmax = fmax
        self.rate = rate
        self.min_duration = min_duration
        self.download = download

        with suppress_output():
            # birdnetlib paramters
            self.analyzer = Analyzer()

        # Define audio filter chain with firequlizer and dynamic audio normalization
        self.audio_filter_chain = audio_filter_chain or ",".join([
            f"aresample=resampler=soxr:out_sample_rate={self.rate}",
            f"firequalizer=gain='if(lt(f,{self.fmin}),-25,if(gt(f,{self.fmax}),-25,0))'",
            "dynaudnorm=b=enabled"
        ])

        # Load audio roi
        self.load_audio_roi()

    def load_audio_roi(self):
        # Build the full filter graph
        if self.download:
            self.audio_filter_chain = f"{self.audio_filter_chain},asplit=2[a1][a2]"

        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-v", "error",
            "-ss", str(self.tmin), "-t", str(self.tmax - self.tmin), "-i", self.path,
            "-filter_complex", self.audio_filter_chain
        ]

        # Map outputs
        if self.download:
            # Define roi path
            filename = os.path.splitext(os.path.basename(self.path))[0]
            output_path = f"{filename}_t_{self.tmin}_{self.tmax}_f_{int(self.fmin)}_{int(self.fmax)}.wav"
            cmd += ["-map", "[a1]", "-f", "f32le", "pipe:1", "-map", "[a2]", "-f", "wav", output_path]
        else:
            cmd += ["-f", "f32le", "pipe:1"]

        # Run FFmpeg and read audio data
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            audio_bytes, error = process.communicate()
            self.buffer = np.frombuffer(audio_bytes, dtype=np.float32)

    def extract_embeddings(self, predict=True, min_confidence=0.5):
        # Split/pad audio rois into 3-second chunks for birdnetlib compatibility
        self.ndarray = self.buffer
        self.sample_secs = 3.0

        self.chunks = []
        for i in range(0, len(self.ndarray), int((self.sample_secs) * self.rate)):
            split = self.ndarray[i : i + int(self.sample_secs * self.rate)]

            # End of signal?
            if len(split) < int(self.min_duration * self.rate):
                break

            # Signal chunk too short? Pad with zeros.
            if len(split) < int(self.rate * self.sample_secs):
                temp = np.zeros((int(self.rate * self.sample_secs)))
                temp[: len(split)] = split
                split = temp

            self.chunks.append(split)

        tmin = self.tmin
        tmax = tmin + self.sample_secs

        results = []
        for sample in self.chunks:
            data = np.array([sample], dtype="float32")
            # Extract embeddings with birdnetlib function
            roi_embeddings = self.analyzer._return_embeddings(data)[0].tolist()
            roi_embeddings = []

            if predict:
                import operator
                pred = self.analyzer.predict(sample)[0]
                p_labels = dict(zip(self.analyzer.labels, pred))
                p_sorted = sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)
                p_sorted = [i for i in p_sorted if i[1] >= min_confidence]

                # Append results
                if len(p_sorted) == 0:
                    results.append({"path": self.path, "tmin": tmin, "tmax": tmax, "fmin": round(self.fmin,2), "fmax": round(self.fmax,2), "embeddings": roi_embeddings, "prediction": "Unknown", "confidence": 0.0})
                else:
                    results.append({"path": self.path, "tmin": tmin, "tmax": tmax, "fmin": round(self.fmin,2), "fmax": round(self.fmax,2), "embeddings": roi_embeddings, "prediction": p_sorted[0][0], "confidence": p_sorted[0][1]})
            else:
                results.append({"path": self.path, "tmin": tmin, "tmax": tmax, "fmin": round(self.fmin,2), "fmax": round(self.fmax,2), "embeddings": roi_embeddings})

            # Increment tmin and tmax
            tmin += self.sample_secs
            tmax = tmin + self.sample_secs

        return results


if __name__ == "__main__":
    # TODO For repertoire analysis noise profile need to be computed on the whole audio file not 3s segments, to be continued
    
    import time
    from tqdm import tqdm

    start_time = time.time()

    # # Set the folder path
    # from pathlib import Path
    # folder_path = Path("Tinamus major")
    # # Iterate over all .wav files in the folder
    # for audio_file in folder_path.glob("*.wav"):
    #     image = ImageROI(path=audio_file, fscale="mel", fmin=150, fmax=8000)
    #     image.detect_rois(min_duration=0.5, display=True)
        
    image = ImageROI(path="SMA04635_20220318_182700.wav", fscale="mel", fmin=150, fmax=8000)
    image.detect_rois(min_duration=1.0, max_duration=10.0, display=False)
    # print(len(image.rois))
    

    roi_embeddings = [] 
    for roi in tqdm(image.rois, desc="Extracting embeddings"):
        audio = AudioROI(
                        path=roi['path'], 
                        tmin=roi['tmin'], tmax=roi['tmax'], 
                        fmin=roi['fmin'], fmax=roi['fmax'], 
                        min_duration=roi['min_duration'], rate=roi['rate'],
                        download=False
                        )
        
        roi_embeddings.append(audio.extract_embeddings(predict=True, min_confidence=0.5))

    # print(roi_embeddings)
    filtered = [item for sublist in roi_embeddings for item in sublist if item.get("prediction") != "Unknown"]
    print(filtered)
                

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")


    # "Turdus_iliacus/XC101610.wav"
    # "SMA04635_20220318_182700.wav"
    
    # "Tinamus major/64136_24.00-27.00.wav"
    # "data/SMA04786_20220113_182200.wav"
    # "Nyctibius bracteatus/5071_3.00-6.00.wav"


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