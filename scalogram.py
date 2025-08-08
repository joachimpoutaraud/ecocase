import os
import subprocess
import numpy as np

import cv2
import librosa

from birdnetlib.main import RecordingBuffer
from birdnetlib.analyzer import Analyzer

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
                start = x1 / self.fps
                end = x2 / self.fps
                fmin = self._y_to_freq[min(int(y2), self.height - 1)]
                fmax = self._y_to_freq[max(int(y1), 0)]

                self.rois.append({
                    "path": self.path,
                    "rate": self.rate,
                    "start": start,
                    "end": end,
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
    def __init__(self, path, start=0.0, end=None, fmin=150, fmax=8000, rate=48000, min_duration=0.5, download=True):
        
        self.path = path
        self.start = start
        self.end = end
        self.fmin = fmin
        self.fmax = fmax
        self.rate = rate
        self.min_duration = min_duration
        self.download = download
        
        # birdnetlib paramters
        self.analyzer = Analyzer()
        self.sample_secs = 3.0
        self.week_48 = -1
        self.date = None
        self.sensitivity = 1.0
        self.lat = None
        self.lon = None
        self.minimum_confidence = 0.1
        self.overlap = 0.0
        self.return_all_detections = False
        self.extracted_audio_paths = {}
        self.extracted_spectrogram_paths = {}
        


    def read_audio_data(self, audio_filter_chain=None, noise_reduction=-25):
        # Define the core audio filter chain
        self.audio_filter_chain = audio_filter_chain or ",".join([
            f"aresample=resampler=soxr:out_sample_rate={self.rate}",
            f"firequalizer=gain='if(lt(f,{self.fmin}),{noise_reduction},if(gt(f,{self.fmax}),{noise_reduction},0))'",
            "dynaudnorm=b=enabled"
        ])

        # Build the full filter graph
        if self.download:
            filter_complex = f"{self.audio_filter_chain},asplit=2[a1][a2]"
        else:
            filter_complex = f"{self.audio_filter_chain}"

        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-v", "error",
            "-ss", str(self.start), "-t", str(self.end - self.start), "-i", self.path,
            "-filter_complex", filter_complex
        ]

        # Define roi path
        filename = os.path.splitext(os.path.basename(self.path))[0]
        self.roi_path = f"{filename}_t_{self.start}_{self.end}_f_{int(self.fmin)}_{int(self.fmax)}.wav"

        # Map outputs
        if self.download:
            cmd += ["-map", "[a1]", "-f", "f32le", "pipe:1", "-map", "[a2]", "-f", "wav", self.roi_path]
        else:
            cmd += ["-f", "f32le", "pipe:1"]

        # Run FFmpeg and read audio data
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            audio_bytes, error = process.communicate()
            self.ndarray = np.frombuffer(audio_bytes, dtype=np.float32)

        # Split audio into 3-second chunks
        self.chunks = []
        for i in range(0, len(self.ndarray), int((self.sample_secs - self.overlap) * self.rate)):
            split = self.ndarray[i : i + int(self.sample_secs * self.rate)]

            # End of signal?
            if len(split) < int(self.min_duration * self.rate):
                break

            # Signal chunk too short? Fill with zeros.
            if len(split) < int(self.rate * self.sample_secs):
                temp = np.zeros((int(self.rate * self.sample_secs)))
                temp[: len(split)] = split
                split = temp

            self.chunks.append(split)

        
        # import librosa
        # import librosa.display
        # import matplotlib.pyplot as plt
        # # Compute the spectrogram
        # D = librosa.amplitude_to_db(librosa.stft(self.ndarray), ref=np.max)
        # # Plot it
        # plt.figure(figsize=(10, 6))
        # librosa.display.specshow(D, sr=self.rate, x_axis='time', y_axis='linear')
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('Spectrogram')
        # plt.tight_layout()
        # plt.show()


if __name__ == "__main__":
    # TODO For repertoire analysis noise profile need to be computed on the whole audio file not 3s segments, to be continued
    
    import time
    from pathlib import Path

    start_time = time.time()

    # # Set the folder path
    # folder_path = Path("Tinamus major")
    # # Iterate over all .wav files in the folder
    # for audio_file in folder_path.glob("*.wav"):
    #     image = ImageROI(path=audio_file, fscale="mel", fmin=150, fmax=8000)
    #     image.detect_rois(min_duration=0.5, display=True)
        
    image = ImageROI(path="out.wav", fscale="mel", fmin=150, fmax=8000)
    image.detect_rois(min_duration=1.0, max_duration=None, display=False)
    # print(len(image.rois))
    
    s = [] 
      
    for roi in image.rois:
        audio = AudioROI(
                        path=roi['path'], 
                        start=roi['start'], end=roi['end'], 
                        fmin=roi['fmin'], fmax=roi['fmax'], 
                        min_duration=roi['min_duration'], rate=roi['rate'],
                        download=False
                        )
        
        # audio.analyze()
        
        # for i, det in enumerate(audio.detections):
        #     if det['confidence'] > 0.5:
        #         s.append(det['scientific_name'])

        audio.extract_embeddings()
        
# # '        print(len(audio.embeddings[0]))

# #         print(audio.embeddings[0])

#         for emb in audio.embeddings_list:
#             print(emb)
#             # Add file path to each dictionary and update start and end time
#             emb['start_time'] = audio.start + emb['start_time']
#             emb['end_time'] = emb['start_time'] + emb['end_time']
#             emb['path'] = audio.path
        
        print(audio.embeddings_list)              



    # s = []
    # audio = AudioROI(
    #                 path="out.wav", 
    #                 start=0.0, end=120, 
    #                 fmin=1, fmax=15000, 
    #                 min_duration=0.5, rate=48000
    #                 )
    
    # audio.analyze()
    # # audio.extract_embeddings()
    
    # for i, det in enumerate(audio.detections):
    #     if det['confidence'] > 0.5:
    #         s.append(det['scientific_name'])


                
    from collections import Counter
    # Count occurrences
    string_counts = Counter(s)
    # Print results
    for string, count in string_counts.items():
        print(f"{string}: {count}")
        
        

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