import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from birdnetlib.main import RecordingBuffer

from rois import extract_roi_contours, extract_roi_chunks
from analysis import analyze_segment


class RecordingROI(RecordingBuffer):
    def __init__(self, path, rate=48000, fmin=1, fmax=15000, audio_filter_chain=None, multiprocessing=True, verbose=True):
        
        self.rate = rate
        self.fmin = fmin
        self.fmax = fmax
        self.multiprocessing = multiprocessing
        self.verbose = verbose

        # Get the file or folder paths
        self.path = []
        if os.path.isdir(path):
            for file in os.listdir(path):
                full_path = os.path.join(path, file)
                if file.lower().endswith((".wav", ".mp3", ".flac", ".aiff", ".ogg")):
                    # Check if audio file is likely valid (> 10 KB)
                    if os.path.getsize(full_path) > 10 * 1024:
                        self.path.append(full_path)
        elif os.path.isfile(path):
            # Check if it's a valid audio file and large enough
            if path.lower().endswith((".wav", ".mp3", ".flac", ".aiff", ".ogg")) and os.path.getsize(path) > 10 * 1024:
                self.path = [path]
        else:
            raise ValueError(f"Invalid path: {path}")
        
        # Get the audio filter chain
        self.audio_filter_chain = audio_filter_chain or ",".join([
            # Resample audio file
            f"aresample=resampler=soxr:out_sample_rate={rate}",
            # Convert to mono
            f"aformat=channel_layouts=mono",
            # Normalize dynamics so embeddings are consistent across files/ROIs
            f"dynaudnorm=b=enabled"
        ])
    
    
    def detect_rois(self, min_duration=0.5, max_duration=10.0, fscale="mel", noise_profile=[0.0, 0.5],
                    n_fft=512, hop_length=128, fps=50, display=True):
        # Set arguments list
        args_list = [(path, self.fmin, self.fmax, self.audio_filter_chain, noise_profile,
                    n_fft, hop_length, fps, min_duration, max_duration, fscale, display)
                    for path in self.path]
        
        # Detect the region of interests (ROIs)
        results = self._multiprocessing(extract_roi_contours, args_list, desc="Detecting region of interests (ROIs)")
        # Flatten the list of rois
        self.rois = [roi for sublist in results if sublist is not None for roi in sublist]

        return self.rois
    

    def segment_rois(self, rois, noise_reduction=-20, download=False):
        # Set arguments list
        args_list = [(roi, self.rate, self.audio_filter_chain, noise_reduction, download) for roi in rois]

        # Segment the region of interests (ROIs)
        results = self._multiprocessing(extract_roi_chunks, args_list, desc="Segmenting region of interests (ROIs)")
        # Flatten the list of segments
        self.segments = [segment for sublist in results if sublist is not None for segment in sublist]      

        return self.segments  

    
    def display_rois(self, embeddings, noise_reduction=-50, frame_width=1400, frame_height=600, point_size=5, port=8080):
        from display import run_bokeh_server
        run_bokeh_server(embeddings, self.rate, self.audio_filter_chain, noise_reduction, frame_width, frame_height, point_size, port)

        
    def birdnet_analysis(self, segments, extract=False, predict=True, min_confidence=0.75):
        # Set arguments list
        args_list = [(seg, self.rate, extract, predict, min_confidence) for seg in segments]

        # Analyze the segments
        results = self._multiprocessing(analyze_segment, args_list, desc="Analyzing segments with BirdNET")
        # Get the results
        self.data = results

        return self.data


    def frequency_analysis(self, predictions, species=None):
        import numpy as np
        # Get unique detected species with counts
        if species is None:
            preds = np.array([item.get('prediction') for item in predictions])
        else:
            filtered_predictions = [d for d in predictions if isinstance(d, dict) and d.get('prediction', '').lower() == species.lower()]
            preds = np.array([item.get('prediction') for item in filtered_predictions])

        unique_species, counts = np.unique(preds, return_counts=True)

        results_dict = {}

        if self.verbose:
            print("--------------------------------------------------")
            print(f"*** Species detections with frequency analysis ***")
            print("--------------------------------------------------")

        for i in np.argsort(-counts):
            species = unique_species[i]
            filtered = [d for d in predictions if d['prediction'] == species]
            confidence = np.array([item.get('confidence') for item in filtered])

            fmin_values = [entry['fmin'] for entry in filtered]
            fmax_values = [entry['fmax'] for entry in filtered]

            fmin_median = np.median(fmin_values) if fmin_values else 0
            fmax_median = np.median(fmax_values) if fmax_values else 0
            avg_confidence = np.nanmean(confidence)

            if self.verbose:
                print(f"{species} ({np.round(fmin_median, 2)} - {np.round(fmax_median,2)} Hz): {counts[i]} detections (avg_confidence={np.round(float(avg_confidence),2)})")

            results_dict[species] = {
                "count": int(counts[i]),
                "fmin_median": float(np.round(fmin_median, 2)),
                "fmax_median": float(np.round(fmax_median, 2)),
                "avg_confidence": float(np.round(avg_confidence, 2))
            }

        return results_dict
    

    def repertoire_analysis(self, embeddings, species, min_samples=3, eps='auto', metric='cosine'):
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics import pairwise_distances

        import hdbscan
        from kneed import KneeLocator

        # Filter embeddings with the selected species
        self.repertoire = [d for d in embeddings if isinstance(d, dict) and d.get('prediction', '').lower() == species.lower()]
        embeddings = np.array([d['embeddings'] for d in self.repertoire if d['embeddings']])

        if eps == 'auto':
            neighbors = NearestNeighbors(n_neighbors=min_samples, metric=metric)
            neighbors_fit = neighbors.fit(embeddings)
            distances, _ = neighbors_fit.kneighbors(embeddings)
            distances = np.sort(distances[:, 1])

            kneedle = KneeLocator(
                x=np.arange(len(distances)),
                y=distances,
                interp_method="polynomial",
                curve="convex",
                direction="increasing"
            )

            try:
                e = float(kneedle.knee_y)
            except:
                e = float(distances.max())
        else:
            e = float(eps)

        # Compute pairwise distances
        precomputed_distances = pairwise_distances(embeddings, metric=metric)
        # Clustering with HDBSCAN
        cluster = hdbscan.HDBSCAN(min_cluster_size=min_samples, cluster_selection_epsilon=e, metric='precomputed').fit(precomputed_distances)
        labels = cluster.labels_
        n_clusters = len(np.unique(labels[labels >= 0]))

        if self.verbose:
            print(f'Number of clusters found (excluding noise): {n_clusters} (min_samples={min_samples}, eps={e})')

        # Update prediction keys with cluster label
        for i, label in enumerate(labels):
            if i < len(self.repertoire):
                self.repertoire[i]['prediction'] = f"Syllable {label}" if label != -1 else "Noise"

        return self.repertoire


    def _multiprocessing(self, func, args_list, desc="Multiprocessing"):
        if self.multiprocessing:
            with Pool(processes=cpu_count()-1) as pool:
                results = list(tqdm(pool.imap(func, args_list),
                                    total=len(args_list),
                                    desc=desc,
                                    disable=not self.verbose))
        else:
            results = [func(args) for args in tqdm(args_list, desc=desc, disable=not self.verbose)]
        return results
        

if __name__ == "__main__":

    ### Create RecordingROI instance
    rec = RecordingROI(path="out.wav", rate=48000, fmin=150, fmax=15000, multiprocessing=True)

    # Detect region of interests (ROIs) in the recording
    rois = rec.detect_rois(min_duration=0.5, max_duration=10.0, display=False)
    # Segment ROIs into 3-second audio chunks
    segments = rec.segment_rois(rois, noise_reduction=-20, download=False)


    # Extract embeddings and predict species with BirdNET
    data = rec.birdnet_analysis(segments, extract=True, predict=True, min_confidence=0.75)
    # Analyze frequency range for each detected species
    results = rec.frequency_analysis(data)
    # Analyze the repertoire of the species
    # data = rec.repertoire_analysis(data, species='Turdus iliacus_Redwing')
    # Display repertoire in interactive plot
    rec.display_rois(data, noise_reduction=-50)




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