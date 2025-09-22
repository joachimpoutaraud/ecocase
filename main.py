import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from birdnetlib.main import RecordingBuffer

from rois import extract_roi_contours, extract_roi_chunks
from analysis import analyze_segment

import warnings
warnings.filterwarnings("ignore")


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


    def birdnet_analysis(self, segments, extract=False, predict=True, min_confidence=0.75, keep_best=True, save_path=None):
        """
        Analyze audio segments using BirdNET and optionally save results as a .npy file.

        Parameters:
            segments (list): List of segment dictionaries.
            extract (bool): Whether to extract audio before prediction.
            predict (bool): Whether to run BirdNET prediction.
            min_confidence (float): Minimum confidence threshold for predictions.
            keep_best (bool): Whether to keep only one embedding with the best confidence score for ROI longer than 3s.
            save_path (str): Path to save the .npy results file. Default is None (no saving).

        Returns:
            list: Analysis results.
        """
        # Set arguments list
        args_list = [(seg, self.rate, extract, predict, min_confidence) for seg in segments]

        # Analyze the segments
        results = self._multiprocessing(analyze_segment, args_list, desc="Analyzing segments with BirdNET")

        if predict and keep_best:
            # Keep embedding with best confidence score for ROIs longer than 3s
            dicts = [r for r in results if isinstance(r, dict)]
            if not dicts:
                self.data = []
            else:
                filtered = []
                current_group = [dicts[0]]

            for prev, curr in tqdm(zip(dicts, dicts[1:]),
                                   total=max(len(dicts)-1, 0),
                                   desc="Filtering embeddings (keeping best per ROI)",
                                   disable=not self.verbose):
                if curr.get('idx') == prev.get('idx') + 1:
                    current_group.append(curr)
                else:
                    best = max(current_group, key=lambda x: x.get('confidence', 0))
                    filtered.append(best)
                    current_group = [curr]

            # Add last group
            best = max(current_group, key=lambda x: x.get('confidence', 0))
            filtered.append(best)
            self.data = filtered
        else:
            # Store results
            self.data = results

        # Save to .npy file if requested
        if save_path is not None:
            import numpy as np
            np.save(save_path, results)

        return self.data


    def frequency_analysis(self, predictions, species=None):
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
    

    def repertoire_analysis(self, embeddings, species, keep_noise=False, random_state=2025, param_dist=None, n_iter_search=20):
        import umap
        import hdbscan
        from sklearn.metrics import make_scorer
        from sklearn.model_selection import RandomizedSearchCV

        # Filter embeddings with the selected species
        self.repertoire = [d for d in embeddings if isinstance(d, dict) and d.get('prediction', '').lower() == species.lower()]
        embeddings = np.array([d['embeddings'] for d in self.repertoire if d.get('embeddings') is not None])

        # Apply UMAP dimensionality reduction
        reducer = umap.UMAP(random_state=random_state)
        reduced_embeddings = reducer.fit_transform(embeddings)
        # Fit HDBSCAN clustering algorithm
        cluster = hdbscan.HDBSCAN(gen_min_span_tree=True).fit(reduced_embeddings)

        # Specify parameters and distributions to sample from
        if param_dist is None:
            param_dist = {
                        'min_samples': [5,10,15,20,25],
                        'min_cluster_size': [5,10,15,20,25],  
                        'metric': ['euclidean', 'manhattan'] 
                        }

        # Apply random search to estimate repertoire size
        validity_scorer = make_scorer(hdbscan.validity.validity_index, greater_is_better=True)
        random_search = RandomizedSearchCV(cluster,
                                           param_distributions=param_dist,
                                           n_iter=n_iter_search,
                                           scoring=validity_scorer,
                                           random_state=random_state)

        random_search.fit(reduced_embeddings)

        # Get best clusterer and labels
        best_clusterer = random_search.best_estimator_
        best_labels = best_clusterer.labels_

        if self.verbose:
            n_clusters = len(np.unique(best_labels[best_labels >= 0]))
            print(f'Number of clusters found (excluding noise): {n_clusters} (best Parameters {random_search.best_params_})')
            print(f"DBCV score :{random_search.best_estimator_.relative_validity_}")

        # Update tmin/tmax, embeddings, and prediction
        for i, d in enumerate(self.repertoire):
            if 'tmin_roi' in d:
                d['tmin'] = d['tmin_roi']
            if 'tmax_roi' in d:
                d['tmax'] = d['tmax_roi']
            d['embeddings'] = reduced_embeddings[i]
            if i < len(best_labels):
                label = best_labels[i]
                d['prediction'] = f"Cluster {label}" if label != -1 else "Noise"

        if keep_noise is not True:
            # Remove all entries labeled as "Noise"
            self.repertoire = [d for d in self.repertoire if d.get('prediction') != "Noise"]

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
    rec = RecordingROI(path="Chloris_chloris", rate=48000, fmin=1, fmax=15000, multiprocessing=True)

    # Detect region of interests (ROIs) in the recording
    rois = rec.detect_rois(min_duration=0.5, max_duration=10.0, display=False)
    # Segment ROIs into 3-second audio chunks
    segments = rec.segment_rois(rois, noise_reduction=-20)

    # Extract embeddings and predict species with BirdNET
    data = rec.birdnet_analysis(segments, extract=True, predict=True, min_confidence=0.25, save_path="chloris.npy")

    # import numpy as np
    # data = np.load('chloris.npy', allow_pickle=True)

    # Analyze frequency range of the species
    results = rec.frequency_analysis(data, species='Chloris chloris_European Greenfinch') # Chloris chloris_European Greenfinch, Turdus iliacus_Redwing
    # Analyze repertoire of the species
    data = rec.repertoire_analysis(data, species='Chloris chloris_European Greenfinch') 
    
    # Display rois in interactive plot
    rec.display_rois(data, noise_reduction=-20)