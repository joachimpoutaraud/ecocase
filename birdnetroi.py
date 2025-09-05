import os, sys, contextlib
import subprocess
import numpy as np

import cv2

from birdnetlib.main import RecordingBuffer
from birdnetlib.analyzer import Analyzer

from image import convert_audio_to_image, convert_pixel_to_frequency


# Suppress stdout/stderr from birdnetlib
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

class BirdNETROI(RecordingBuffer):
    def __init__(self, path, rate=48000, fmin=1, fmax=15000, audio_filter_chain=None):
        
        self.path = path
        self.rate = rate
        self.fmin = fmin
        self.fmax = fmax

        self.audio_filter_chain = audio_filter_chain or ",".join([
            # Keep sample rate stable and convert to mono
            f"aresample=resampler=soxr:out_sample_rate={rate},aformat=channel_layouts=mono",
            # Normalize dynamics so embeddings are consistent across files/ROIs
            f"dynaudnorm=b=enabled"
        ])

        with suppress_output():
            # birdnetlib paramters
            self.analyzer = Analyzer()

    def detect_rois(self, min_duration=0.5, max_duration=10.0, fscale="mel", noise_profile=[0.0, 0.5], n_fft=512, hop_length=128, fps=50, display=True):
        # Convert audio to image
        image, height, width = convert_audio_to_image(self.path, self.rate, self.fmin, self.fmax, self.audio_filter_chain, noise_profile, n_fft, hop_length, fps)
        converted_image, px_to_freq = convert_pixel_to_frequency(image, self.fmin, self.fmax, height, fscale)

        try:      
            # Postprocess gray image for ROI analysis   
            filtered = cv2.bilateralFilter(converted_image, d=9, sigmaColor=75, sigmaSpace=75)
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
                duration_roi = w / fps
                
                # Skip ROIs that are shorter than min_duration and, optionally, longer than max_duration
                if duration_roi < min_duration or (max_duration is not None and duration_roi > max_duration):
                    continue
                  
                # Skip ROIs that span the entire image (both width and height)
                if w >= width * 0.98 and h >= height * 0.98:
                    continue
                
                boxes.append((x, y, x + w, y + h))

            self.rois = []

            for x1, y1, x2, y2 in boxes:
                tmin = x1 / fps
                tmax = x2 / fps
                fmin = px_to_freq[min(int(y2), height - 1)]
                fmax = px_to_freq[max(int(y1), 0)]

                self.rois.append({
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
                ax.imshow(converted_image, cmap="gray")
                for roi in self.rois:
                    x1, y1, x2, y2 = roi["bounds"]
                    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                            linewidth=2, edgecolor='lime', facecolor='none')
                    ax.add_patch(rect)
                ax.set_title("Detected ROIs with Bounding Boxes")
                plt.show()

            return self.rois

        except Exception as e:
            print(f"Failed to extract ROIs: {e}")
            return []
        
    def segment_rois(self, rois, noise_reduction=-20, download=False):
        self.segments = []
        # Process audio rois into 3-second segments for birdnetlib compatibility
        self.sample_secs = 3.0 
        total_samples = int(self.sample_secs * self.rate)

        # Iterate over each ROI and segment audio
        for roi in tqdm(rois, desc="Segment rois"):
            # Add firequalizer to audio filter chain to isolate frequency band of interest
            audio_filter_chain = f"{self.audio_filter_chain},firequalizer=gain='if(between(f,{int(roi['fmin'])},{int(roi['fmax'])}),0,{noise_reduction})'"

            if download:
                audio_filter_chain = f"{audio_filter_chain},asplit=2[a1][a2]"

            # Build FFmpeg command
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-v", "error",
                "-ss", str(roi['tmin']), "-t", str(roi['tmax'] - roi['tmin']), "-i", self.path,
                "-filter_complex", audio_filter_chain
            ]

            # Map outputs
            if download:
                # Define roi path and output folder
                filename = os.path.splitext(os.path.basename(self.path))[0]
                output_folder = "rois"
                os.makedirs(output_folder, exist_ok=True)
                filename_roi = f"{filename}_t_{roi['tmin']}_{roi['tmax']}_f_{int(roi['fmin'])}_{int(roi['fmax'])}.wav"
                self.output_path = os.path.join(output_folder, filename_roi)
                cmd += ["-map", "[a1]", "-f", "f32le", "pipe:1", "-map", "[a2]", "-f", "wav", self.output_path]
            else:
                cmd += ["-f", "f32le", "pipe:1"]

            # Run FFmpeg and read audio data
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
                audio_bytes, error = process.communicate()
                self.ndarray = np.frombuffer(audio_bytes, dtype=np.float32)

            for i in range(0, len(self.ndarray), total_samples):
                split = self.ndarray[i : i + total_samples]

                # End of signal?
                if len(split) < int(roi['min_duration'] * self.rate):
                    break

                # Default start index for this chunk (relative to the extracted ROI audio)
                start_idx = i
                # Signal chunk too short? Center the split using audio file
                if len(split) < total_samples:
                    # Find the center position in the original audio
                    start_idx = max(0, i - (total_samples - len(split)) // 2)
                    # Clip indices to valid range
                    start_idx = max(0, min(start_idx, len(self.ndarray) - total_samples))
                    end_idx = start_idx + total_samples
                    split = self.ndarray[start_idx:end_idx]
                    # If still not enough samples (e.g. entire file shorter than total_samples), pad with zeros
                    if len(split) < total_samples:
                        temp = np.zeros(total_samples, dtype=self.ndarray.dtype)
                        temp[:len(split)] = split
                        split = temp

                # Extend segments list along with its start index (relative to the ROI audio)
                self.segments.append({"samples": split, "idx": start_idx, "tmin": roi['tmin'], "fmin": roi['fmin'], "fmax": roi['fmax']})

        return self.segments


    def extract_embeddings(self, segments, predict=True, min_confidence=0.5):
        self.emb = []

        for seg in tqdm(segments, desc="Extract embeddings"):
            c = seg["samples"]

            # compute tmin/tmax for this seg using the stored start index
            tmin_seg = seg["tmin"] + (seg["idx"] / float(self.rate))
            tmax_seg = tmin_seg + self.sample_secs

            # Extract embeddings with birdnetlib function
            e = self.analyzer._return_embeddings(np.array([c], dtype="float32"))[0].tolist()

            if predict:
                import operator
                pred = self.analyzer.predict(c)[0]
                # Assign scores to labels
                p_labels = dict(zip(self.analyzer.labels, pred))
                # Sort by score
                p_sorted = sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)
                # Filter by minimum confidence
                p_sorted = [i for i in p_sorted if i[1] >= min_confidence]

                # Append results
                if len(p_sorted) == 0:
                    self.emb.append({
                        "path": self.path,
                        "tmin": tmin_seg,
                        "tmax": tmax_seg,
                        "fmin": seg["fmin"],
                        "fmax": seg["fmin"],
                        "embeddings": e,
                        "prediction": "Unknown",
                        "confidence": 0.0
                    })
                else:
                    self.emb.append({
                        "path": self.path,
                        "tmin": tmin_seg,
                        "tmax": tmax_seg,
                        "fmin": seg["fmin"],
                        "fmax": seg["fmin"],
                        "embeddings": e,
                        "prediction": p_sorted[0][0],
                        "confidence": p_sorted[0][1]
                    })
            else:
                self.emb.append({
                    "path": self.path,
                    "tmin": tmin_seg,
                    "tmax": tmax_seg,
                    "fmin": seg["fmin"],
                    "fmax": seg["fmin"],
                    "embeddings": e
                })
        
        return self.emb
        

if __name__ == "__main__":
    # TODO For repertoire analysis noise profile need to be computed on the whole audio file not 3s segments, to be continued

    from tqdm import tqdm
    from display import display_rois

    # # Set the folder path
    # from pathlib import Path
    # folder_path = Path("test")
    # # Iterate over all .wav files in the folder
    # rois = []
    # for audio_file in sorted(folder_path.iterdir()):
    #     if audio_file.suffix.lower() not in {'.wav', '.mp3', '.flac'}:
    #         continue
    #     print(audio_file)
    #     image = ImageROI(path=str(audio_file), fscale="mel", fmin=1, fmax=15000)
    #     image.detect_rois(min_duration=0.5, max_duration=10.0, display=False)
    #     rois.extend(image.rois)
        
    ### DETECT ROIS ON SPECTROGRAM IMAGE
    birdroi = BirdNETROI(path="out.wav", rate=48000, fmin=150, fmax=15000)

    birdroi.detect_rois(min_duration=0.5, max_duration=10.0, display=False)
    # Segment the region of interests
    birdroi.segment_rois(birdroi.rois, noise_reduction=-20, download=True)
    # Extract embeddings from BirdNET and predict species
    birdroi.extract_embeddings(birdroi.segments, min_confidence=0.75)

    ### DISPLAY ROIS
    display_rois(birdroi.emb, noise_reduction=-50)


    # ### FREQUENCY ANALYSIS
    # # Flatten list of lists
    # embeddings = [item for sublist in embeddings for item in sublist] 
    # # Get unique detected species with counts
    # predictions = np.array([item.get('prediction') for item in embeddings])
    # unique_species, counts = np.unique(predictions, return_counts=True)
    # print(f"\n*** Species detections with frequency range ***")
    # print("-----------------------------------------------")

    # for i in np.argsort(-counts):
    #     # Filter embeddings for a specific species
    #     filtered = [d for d in embeddings if d['prediction'] == unique_species[i]]
    #     confidence = np.array([item.get('confidence') for item in filtered])
    #     # Collect fmin and fmax values
    #     fmin_values = []
    #     fmax_values = []

    #     for entry in filtered:
    #         fmin_values.append(entry['fmin'])
    #         fmax_values.append(entry['fmax'])

    #     # Calculate frequency range
    #     fmin_median = np.median(fmin_values) if fmin_values else 0
    #     fmax_median = np.median(fmax_values) if fmin_values else 0
    #     avg_confidence = np.nanmean(confidence)
    #     print(f"{unique_species[i]} ({np.round(fmin_median,2)} - {np.round(fmax_median,2)} Hz): {counts[i]} detections (avg_confidence={np.round(float(avg_confidence),2)})")


    ### REPERTOIRE ANALYSIS



    # ### CLUSTER ROIS AND PREDICT 
    # from clustering import ClusterROI   

    # predictions = np.array([item['prediction'] for item in embeddings])
    # confidence = np.array([item['confidence'] for item in embeddings])

    # clusterer = ClusterROI(embeddings)
    # clusterer.cluster_rois(predictions)
    # # clusterer.predict_cluster(label=-1)

    # for label in np.unique(clusterer.labels):
    #     if label == None:
    #         continue

    #     cluster_predictions = predictions[clusterer.labels == label]
    #     cluster_confidence = confidence[clusterer.labels == label]

    #     # Count frequency of each species
    #     unique_species, counts = np.unique(cluster_predictions, return_counts=True)
    #     most_frequent_species = unique_species[np.argmax(counts)]

    #     if most_frequent_species == "Unknown":
    #         print(f"Label {label} ({len(cluster_predictions)} ROIs): Prediction → Unknown, Confidence not considered.")
    #         continue

    #     # Compute weighted score for each species
    #     species_scores = {}
    #     for species in np.unique(cluster_predictions):
    #         mask = cluster_predictions == species
    #         avg_conf = np.mean(cluster_confidence[mask])
    #         freq = np.sum(mask)
    #         score = avg_conf * np.log(freq + 1) # Confidence-weighted frequency score
    #         species_scores[species] = (score, avg_conf, freq)

    #     # Select species with highest score
    #     best_species, (best_score, best_conf, best_freq) = max(species_scores.items(), key=lambda x: x[1][0])
    #     print(f"Label {label} ({len(cluster_predictions)} ROIs): Prediction → {best_species}, Confidence → {best_conf:.2f}")



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

# # Folder with audio segs
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

# # Calculate overall mean values
# if fmin_list and fmax_list:
#     mean_fmin = np.median(fmin_list)
#     mean_fmax = np.median(fmax_list)
#     print(f"\nMean fmin: {mean_fmin:.2f} Hz")
#     print(f"Mean fmax: {mean_fmax:.2f} Hz")
# else:
#     print("\nNo pitch data available to compute mean values.")