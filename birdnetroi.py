import os, sys, contextlib
import numpy as np

import cv2
from tqdm import tqdm

from birdnetlib.main import RecordingBuffer
from birdnetlib.analyzer import Analyzer

from utils import convert_audio_to_image, convert_pixel_to_frequency, extract_roi


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

class RecordingROI(RecordingBuffer):
    def __init__(self, path, rate=48000, fmin=1, fmax=15000, audio_filter_chain=None):
        
        self.path = path
        self.rate = rate
        self.fmin = fmin
        self.fmax = fmax

        self.audio_filter_chain = audio_filter_chain or ",".join([
            # Resample audio file
            f"aresample=resampler=soxr:out_sample_rate={rate}",
            # Convert to mono
            f"aformat=channel_layouts=mono",
            # Normalize dynamics so embeddings are consistent across files/ROIs
            f"dynaudnorm=b=enabled"
        ])

        with suppress_output():
            # birdnetlib paramters
            self.analyzer = Analyzer()


    def detect_rois(self, min_duration=0.5, max_duration=10.0, fscale="mel", noise_profile=[0.0, 0.5], n_fft=512, hop_length=128, fps=50, display=True):
        # Convert audio to image
        image, height, width = convert_audio_to_image(self.path, self.fmin, self.fmax, self.audio_filter_chain, noise_profile, n_fft, hop_length, fps)
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
                    "path": self.path,
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
        total_samples = int(3.0 * self.rate)

        # Iterate over each ROI and segment audio
        for roi in tqdm(rois, desc="Segmenting rois"):
            # Extract ROI audio using utils function
            self.ndarray = extract_roi(roi['path'], roi['tmin'], roi['tmax'], roi['fmin'], roi['fmax'], self.audio_filter_chain, noise_reduction, download, rate=self.rate)[0]

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
                self.segments.append({"samples": split, "idx": start_idx, "path": roi['path'], "tmin": roi['tmin'], "fmin": roi['fmin'], "fmax": roi['fmax']})

        return self.segments


    def birdnet_analysis(self, segments, extract=False, predict=True, min_confidence=0.75):
        from operator import itemgetter
        import librosa

        self.data = []
        birdnet_sr = 48000

        def build_result(seg, tmin, tmax, embeddings=None, prediction="Unknown", confidence=0.0):
            result = {
                "path": seg["path"],
                "tmin": tmin,
                "tmax": tmax,
                "fmin": seg["fmin"],
                "fmax": seg["fmax"],
            }
            if extract:
                result["embeddings"] = embeddings

            result["prediction"] = prediction
            result["confidence"] = confidence
            return result

        for seg in tqdm(segments, desc="Processing segments with BirdNET"):
            samples = seg["samples"]
            if self.rate != birdnet_sr:
                samples = librosa.resample(samples, orig_sr=self.rate, target_sr=birdnet_sr)

            tmin_seg = seg["tmin"] + (seg["idx"] / birdnet_sr)
            tmax_seg = tmin_seg + 3.0  # BirdNET chunk duration

            embeddings = None
            prediction = "Unknown"
            confidence = 0.0

            if extract:
                embeddings = self.analyzer._return_embeddings([samples])[0].tolist()

            if predict:
                predictions = self.analyzer.predict(samples)[0]
                label_scores = dict(zip(self.analyzer.labels, predictions))
                sorted_labels = sorted(label_scores.items(), key=itemgetter(1), reverse=True)
                filtered = [item for item in sorted_labels if item[1] >= min_confidence]

                if filtered:
                    prediction, confidence = filtered[0]

            result = build_result(seg, tmin_seg, tmax_seg, embeddings, prediction, confidence)
            self.data.append(result)

        return self.data


    def frequency_analysis(self, predictions):
        # Get unique detected species with counts
        preds = np.array([item.get('prediction') for item in predictions])
        unique_species, counts = np.unique(preds, return_counts=True)
        print("--------------------------------------------------")
        print(f"*** Species detections with frequency analysis ***")
        print("--------------------------------------------------")

        for i in np.argsort(-counts):
            # Filter embeddings for a specific species
            filtered = [d for d in predictions if d['prediction'] == unique_species[i]]
            confidence = np.array([item.get('confidence') for item in filtered])
            # Collect fmin and fmax values
            fmin_values = []
            fmax_values = []

            for entry in filtered:
                fmin_values.append(entry['fmin'])
                fmax_values.append(entry['fmax'])

            # Calculate frequency range
            fmin_median = np.median(fmin_values) if fmin_values else 0
            fmax_median = np.median(fmax_values) if fmin_values else 0
            avg_confidence = np.nanmean(confidence)
            print(f"{unique_species[i]} ({np.round(fmin_median, 2)} - {np.round(fmax_median,2)} Hz): {counts[i]} detections (avg_confidence={np.round(float(avg_confidence),2)})")

    
    def display_rois(self, embeddings, noise_reduction=-50, frame_width=1400, frame_height=600, point_size=5, port=8080):
        from display import run_bokeh_server
        run_bokeh_server(embeddings, self.rate, self.audio_filter_chain, noise_reduction, frame_width, frame_height, point_size, port)
        

if __name__ == "__main__":
    # TODO For repertoire analysis noise profile need to be computed on the whole audio file not 3s segments, to be continued

    # # Set the folder path
    # from pathlib import Path
    # folder_path = Path("test")
    # # Iterate over all .wav files in the folder
    # rois = []
    # for audio_file in sorted(folder_path.iterdir()):
    #     if audio_file.suffix.lower() not in {'.wav', '.mp3', '.flac'}:
    #         continue
    #     print(audio_file)
    #     rec = RecordingROI(path=str(audio_file), rate=48000, fmin=1, fmax=15000)
    #     rec.detect_rois(min_duration=0.5, max_duration=10.0, display=False)
    #     rois.extend(rec.rois)

    ### Create RecordingROI instance
    rec = RecordingROI(path="out.wav", rate=48000, fmin=150, fmax=15000)

    # Detect region of interests (ROIs) in the recording
    rois = rec.detect_rois(min_duration=0.5, max_duration=10.0, display=False)
    # Segment ROIs into 3-second audio clips
    segments = rec.segment_rois(rois, noise_reduction=-20, download=False)

    # Predict species with BirdNET
    predictions = rec.birdnet_analysis(segments, predict=True, min_confidence=0.75)
    # Analyze frequency range for each detected species
    rec.frequency_analysis(predictions)

    # # Extract embeddings and predict species with BirdNET
    # data = rec.birdnet_analysis(segments, extract=True, predict=True, min_confidence=0.75)
    # # Display ROIs in interactive plot
    # rec.display_rois(data, noise_reduction=-50)




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