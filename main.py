import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from birdnetlib.main import RecordingBuffer

from rois import extract_roi_contours, extract_roi_audio
from analysis import prepare_segments, analyze_segments
from display import run_bokeh_server

import warnings
warnings.filterwarnings("ignore")


class RecordingROI(RecordingBuffer):
    def __init__(self, path, rate=48000, fmin=1, fmax=15000, audio_filter_chain=None, multiprocessing=True, save_path=None, verbose=True):
        
        self.rate = rate
        self.fmin = fmin
        self.fmax = fmax
        self.multiprocessing = multiprocessing
        self.save_path = save_path
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
    

    def segment_rois(self, rois, noise_reduction=-20, download=False, output_folder="rois"):
        # Set arguments list
        args_list = [(roi['path'], roi['tmin'], roi['tmax'], roi['fmin'], roi['fmax'], self.audio_filter_chain, noise_reduction, download, self.rate) for roi in rois]

        # Segment the region of interests (ROIs)
        self.data = self._multiprocessing(extract_roi_audio, args_list, desc="Segmenting region of interests (ROIs)")

        # Save to .npy file if requested
        if self.save_path is not None:
            np.save(self.save_path, self.data)    

        return self.data  


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
    


class RecordingAnalysis(RecordingROI):
    def __init__(self, data, audio_filter_chain=None, multiprocessing=True, random_state=2025, save_path=None, verbose=True):
        
        self.multiprocessing = multiprocessing
        self.random_state = random_state
        self.save_path = save_path
        self.verbose = verbose

        try:
            # Get the file data path
            if os.path.isfile(data):
                # Check if it's a valid .npy file
                if data.lower().endswith((".npy")):
                    self.data = np.load(data, allow_pickle=True)
                else:
                    raise ValueError(f"Invalid path: {data}")
        except:
            self.data = data

        if audio_filter_chain is not None:
            self.audio_filter_chain = audio_filter_chain
        else:
            self.audio_filter_chain = self.data[0].get('audio_filter_chain', None)
        
        # Set random seed for reproducibility    
        np.random.seed(self.random_state)


    def birdnet_analysis(self, extract=False, predict=True, min_confidence=0.25, keep_best=True):
        """
        Analyze audio segments using BirdNET and optionally save results as a .npy file.

        Parameters:
            data (list): List of segment dictionaries.
            extract (bool): Whether to extract audio before prediction.
            predict (bool): Whether to run BirdNET prediction.
            min_confidence (float): Minimum confidence threshold for predictions.
            keep_best (bool): Whether to keep only one embedding with the best confidence score for ROI longer than 3s.
            save_path (str): Path to save the .npy results file. Default is None (no saving).

        Returns:
            list: Analysis results.
        """

        # Set arguments list
        args_list = [roi for roi in self.data]
        # Segment the region of interests (ROIs)
        results = self._multiprocessing(prepare_segments, args_list, desc="Preparing segments for BirdNET analysis")
        # Flatten the list of segments
        segments = [segment for sublist in results if sublist is not None for segment in sublist]  
        # Set arguments list
        args_list = [(seg, extract, predict, min_confidence) for seg in segments]
        # Analyze the segments
        results = self._multiprocessing(analyze_segments, args_list, desc="Analyzing segments with BirdNET")

        if predict and keep_best:
            # Keep embedding with best confidence score for ROIs longer than 3s
            dicts = [r for r in results if isinstance(r, dict)]
            if not dicts:
                self.data = []
            else:
                filtered = []
                current_group = [dicts[0]]

            for prev, curr in zip(dicts, dicts[1:]):
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

        elif keep_best is False:
            # Store results and update any dict entries to use chunk boundaries
            for d in results:
                if isinstance(d, dict):
                    if 'tmin_chunk' in d:
                        d['tmin'] = d['tmin_chunk']
                    if 'tmax_chunk' in d:
                        d['tmax'] = d['tmax_chunk']
            self.data = results

        else:
            self.data = results

        # Save to .npy file if requested
        if self.save_path is not None:
            np.save(self.save_path, self.data)

        return self.data


    def frequency_analysis(self, species=None):
        # Get unique detected species with counts
        if species is None:
            preds = np.array([item.get('prediction') for item in self.data])
        else:
            filtered_predictions = [d for d in self.data if isinstance(d, dict) and d.get('prediction', '').lower() == species.lower()]
            preds = np.array([item.get('prediction') for item in filtered_predictions])

        unique_species, counts = np.unique(preds, return_counts=True)

        results_dict = {}

        if self.verbose:
            print("--------------------------------------------------")
            print(f"*** Species detections with frequency analysis ***")
            print("--------------------------------------------------")

        for i in np.argsort(-counts):
            species = unique_species[i]
            filtered = [d for d in self.data if d['prediction'] == species]
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
    

    def repertoire_analysis(self, species, keep_noise=False, param_dist=None, n_iter_search=20):
        import umap
        import hdbscan
        from sklearn.metrics import make_scorer
        from sklearn.model_selection import RandomizedSearchCV

        # Filter embeddings with the selected species
        self.data = [d for d in self.data if isinstance(d, dict) and d.get('prediction', '').lower() == species.lower()]
        embeddings = np.array([d['embeddings'] for d in self.data if d.get('embeddings') is not None])

        if embeddings.shape[1] > 2:
            # Apply UMAP dimensionality reduction
            reducer = umap.UMAP(random_state=self.random_state)
            reduced_embeddings = reducer.fit_transform(embeddings)
        else:
            reduced_embeddings = embeddings
            
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
                                           random_state=self.random_state)
        # Fit random search algorithm
        random_search.fit(reduced_embeddings)

        # Get best clusterer and labels
        best_clusterer = random_search.best_estimator_
        best_labels = best_clusterer.labels_

        if self.verbose:
            n_clusters = len(np.unique(best_labels[best_labels >= 0]))
            print(f'Number of clusters found (excluding noise): {n_clusters} (best Parameters {random_search.best_params_})')
            print(f"DBCV score :{random_search.best_estimator_.relative_validity_}")

        # Update embeddings and add 'repertoire' key with label values
        for i, d in enumerate(self.data):
            d['embeddings'] = reduced_embeddings[i]
            if i < len(best_labels):
                label = best_labels[i]
                d['repertoire'] = f"Cluster {label}" if label != -1 else "Noise"

        if keep_noise is not True:
            # Remove all entries labeled as "Noise"
            self.data = [d for d in self.data if d.get('repertoire') != "Noise"]
            
        # Save to .npy file if requested
        if self.save_path is not None:
            np.save(self.save_path, self.data)

        return self.data
    
    
    def display_rois(self, noise_reduction=-20, frame_width=1400, frame_height=600, point_size=5, port=8080):
        # Display rois in interactive plot with Bokeh server
        run_bokeh_server(self.data, self.audio_filter_chain, noise_reduction, frame_width, frame_height, point_size, port, self.random_state)

        

if __name__ == "__main__":
    ### Create RecordingROI instance
    rec = RecordingROI(path='Turdus_iliacus', multiprocessing=True, save_path='rois.npy')
    # Detect region of interests (ROIs) in the recording
    rois = rec.detect_rois(min_duration=0.5, max_duration=10.0, display=False)
    # Segment region of interests (ROIs) in the recording
    data = rec.segment_rois(rois, noise_reduction=-20)


    ### Create RecordingAnalysis instance
    rec = RecordingAnalysis(data=data, multiprocessing=True)
    # Extract embeddings and predict species with BirdNET
    embeddings = rec.birdnet_analysis(extract=True, predict=True, min_confidence=0.25)
    # Analyze frequency range of the predicted species
    results = rec.frequency_analysis(species='Turdus iliacus_Redwing') 
    # Analyze repertoire of the species
    repertoire = rec.repertoire_analysis(species='Turdus iliacus_Redwing') # Chloris chloris_European Greenfinch, Turdus iliacus_Redwing
    
    # Display rois in interactive plot
    rec.display_rois(noise_reduction=-20)



    #TODO
    # 1) system to choose from the labels species list