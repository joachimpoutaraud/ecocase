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


class EcoCASE(RecordingBuffer):
    def __init__(self, path, rate=48000, fmin=1, fmax=15000, audio_filter_chain=None, multiprocessing=True, verbose=True, random_state=2025):
        
        self.rate = rate
        self.fmin = fmin
        self.fmax = fmax
        self.multiprocessing = multiprocessing
        self.verbose = verbose
        self.random_state = random_state

        # Set random seed for reproducibility    
        np.random.seed(self.random_state)

        # Load path(s) or .npy data
        self._load_path(path)
        
        # Get the audio filter chain
        self.audio_filter_chain = audio_filter_chain or ",".join([
            # Resample audio file
            f"aresample=resampler=soxr:out_sample_rate={rate}",
            # Convert to mono
            f"aformat=channel_layouts=mono",
            # Normalize dynamics
            f"dynaudnorm=b=enabled",
        ])
    

    def detect_rois(self, min_duration=0.5, max_duration=10.0, fscale="mel", noise_profile=[0.0, 0.5],
                    n_fft=512, hop_length=128, fps=50, stretch_factor=1.0, display=True):
        """
        Detect region of interests (ROIs) from audio files.
        
        Parameters:
            min_duration (float): Minimum duration of ROIs in seconds.
            max_duration (float): Maximum duration of ROIs in seconds.
            fscale (str): Frequency scale for spectrogram ('mel', 'log', 'linear').
            noise_profile (list): Noise profile for denoising audio file with FFT [min, max] (seconds).
            n_fft (int): Number of FFT components for spectrogram.
            hop_length (int): Hop length for STFT.
            fps (int): Frames per second for spectrogram.
            stretch_factor (float): Time stretch factor for contour detection.
            display (bool): Whether to display detected ROIs in a plot.
            
            Returns:
            rois (list): List of detected ROIs dictionaries.
        """
        
        # Set arguments list
        args_list = [(path, self.fmin, self.fmax, self.audio_filter_chain, noise_profile,
                    n_fft, hop_length, fps, min_duration, max_duration, fscale, stretch_factor, display)
                    for path in self.path]
        
        # Detect the region of interests (ROIs)
        results = self._multiprocessing(extract_roi_contours, args_list, desc="Detecting region of interests (ROIs)")
        # Flatten the list of rois
        rois = [roi for sublist in results if sublist is not None for roi in sublist]

        return rois
    

    def segment_rois(self, rois, noise_reduction=-20, download=False, binary_file_path=None):
        """
        Segment region of interests (ROIs) from audio files and optionally save results as a .npy file.
        
        Parameters:
            rois (list): List of ROIs dictionaries.
            noise_reduction (int): Noise reduction level in dB.
            download (bool): Whether to save the segmented ROIs as audio files.
            binary_file_path (str): Path to save the .npy results file. Default is None (no saving).
            
            Returns:
            rois (list): List of segmented ROIs data.
        """

        # Set arguments list
        args_list = [(roi['path'], roi['tmin'], roi['tmax'], roi['fmin'], roi['fmax'], self.audio_filter_chain, noise_reduction, download, self.rate) for roi in rois]

        # Segment the region of interests (ROIs)
        self.rois = self._multiprocessing(extract_roi_audio, args_list, desc="Segmenting region of interests (ROIs)")

        # Save rois to a binary file in NumPy .npy format
        if binary_file_path is not None:
            np.save(binary_file_path, self.rois)    

        return self.rois  


    def birdnet_analysis(self, rois, embeddings=False, predict=True, min_confidence=0.25, keep_best=True, binary_file_path=None):
        """
        Analyze region of interests (ROIs) using BirdNET and optionally save results as a .npy file.

        Parameters:
            rois (list): List of ROIs dictionaries.
            embeddings (bool): Whether to extract BirdNET embeddings or not.
            predict (bool): Whether to run BirdNET prediction or not.
            min_confidence (float): Minimum confidence threshold for predictions.
            keep_best (bool): Whether to keep only one embedding with the best confidence score for ROI longer than 3s.
            binary_file_path (str): Path to save the .npy results file. Default is None (no saving).

        Returns:
            data (list): Analysis results.
        """

        # Check if rois is a file path
        if isinstance(rois, str):
            rois = np.load(rois, allow_pickle=True)

        # Set arguments list
        args_list = [roi for roi in rois]
        # Segment the region of interests (ROIs)
        results = self._multiprocessing(prepare_segments, args_list, desc="Preparing segments for BirdNET analysis")
        # Flatten the list of segments
        segments = [segment for sublist in results if sublist is not None for segment in sublist]  
        # Set arguments list
        args_list = [(seg, embeddings, predict, min_confidence) for seg in segments]
        # Analyze the segments
        results = self._multiprocessing(analyze_segments, args_list, desc="Analyzing segments with BirdNET")

        if predict and keep_best:
            # Keep only one embedding (highest confidence) per contiguous idx group
            dicts = [r for r in results if isinstance(r, dict)]
            if not dicts:
                self.data = []
            else:
                # Ensure items are ordered by idx before grouping
                dicts.sort(key=lambda x: x.get('idx', 0))

            filtered = []
            current_group = [dicts[0]]

            for curr in dicts[1:]:
                prev = current_group[-1]
                if curr.get('idx') == prev.get('idx', 0) + 1:
                    current_group.append(curr)
                else:
                    # Finalize previous group by keeping the entry with max confidence
                    best = max(current_group, key=lambda x: x.get('confidence', 0))
                    filtered.append(best)
                    current_group = [curr]

            # Finalize last group
            best = max(current_group, key=lambda x: x.get('confidence', 0))
            filtered.append(best)
            self.data = filtered

        elif keep_best is False:
            # Keep all results if chunk boundaries exist, update tmin/tmax to chunk values
            for d in results:
                if isinstance(d, dict):
                    if 'tmin_chunk' in d:
                        d['tmin'] = d['tmin_chunk']
                    if 'tmax_chunk' in d:
                        d['tmax'] = d['tmax_chunk']
            self.data = results

        else:
            # Default store raw results
            self.data = results

        # Save data to a binary file in NumPy .npy format
        if binary_file_path is not None:
            np.save(binary_file_path, self.data)

        return self.data


    def frequency_analysis(self, species=None):
        """
        Analyze frequency range of detected species in the ROIs.

        Parameters:
            species (str): Specific species to analyze. If None, analyze all detected species.

        Returns:
            results_dict (dict): Dictionary with frequency analysis results.
        """

        # Get unique detected species with counts
        if species is None:
            preds = np.array([item.get('prediction') for item in self.data])
        else:
            filtered_predictions = [d for d in self.data if isinstance(d, dict) and d.get('prediction', '').lower() == species.lower()]
            preds = np.array([item.get('prediction') for item in filtered_predictions])

        unique_species, counts = np.unique(preds, return_counts=True)

        results_dict = {}

        for i in np.argsort(-counts):
            species = unique_species[i]
            filtered = [d for d in self.data if d['prediction'] == species]
            confidence = np.array([item.get('confidence') for item in filtered])

            fmin_values = [entry['fmin'] for entry in filtered]
            fmax_values = [entry['fmax'] for entry in filtered]

            fmin_median = np.median(fmin_values) if fmin_values else 0
            fmax_median = np.median(fmax_values) if fmax_values else 0
            avg_confidence = np.nanmean(confidence)

            results_dict[species] = {
                "fmin_median": float(np.round(fmin_median, 2)),
                "fmax_median": float(np.round(fmax_median, 2)),
                "detections": int(counts[i]),
                "avg_confidence": float(np.round(avg_confidence, 2)),
            }

        return results_dict
    

    def repertoire_analysis(self, species, keep_noise=False, param_dist=None, n_iter_search=20, binary_file_path=None):
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
            
        # Save data to a binary file in NumPy .npy format
        if binary_file_path is not None:
            np.save(binary_file_path, self.data)

        return self.data
    
    
    def display_clusters(self, noise_reduction=-20, frame_width=1400, frame_height=600, point_size=5, port=8080):
        # Display clusters in an interactive plot with Bokeh server
        run_bokeh_server(self.data, self.audio_filter_chain, noise_reduction, frame_width, frame_height, point_size, port, self.random_state)


    def _multiprocessing(self, func, args_list, desc="Multiprocessing"):
        """
        Helper function to run multiprocessing with optional progress bar.
        
        Parameters:
            func (function): Function to apply to each set of arguments.
            args_list (list): List of argument tuples for the function.
            desc (str): Description for the progress bar.
            
        Returns:
            results (list): List of results from applying the function.
        """

        if self.multiprocessing:
            with Pool(processes=cpu_count()-1) as pool:
                results = list(tqdm(pool.imap(func, args_list),
                                    total=len(args_list),
                                    desc=desc,
                                    disable=not self.verbose))
        else:
            results = [func(args) for args in tqdm(args_list, desc=desc, disable=not self.verbose)]
        return results


    def _load_path(self, path, min_size=10*1024, audio_ext=(".wav", ".mp3", ".flac", ".aiff", ".ogg")):
        """
        Load audio file paths into self.path or load binary file data into self.data.

        - If `path` is a directory, walk it and add audio files larger than min_size.
        - If `path` is a file and an audio file, add it if larger than min_size.
        - If `path` is a .npy file, load it into self.data (allow_pickle=True).

        Parameters:
            path (str): Directory, audio file, or .npy file.
            min_size (int): Minimum file size in bytes to consider for audio files. Default is 10 KB.
            audio_ext (tuple): Tuple of audio file extensions to consider.  Default includes common formats.

        Returns:
            list: List of audio file paths loaded into self.path.
        """

        self.path = []

        if isinstance(path, (list, tuple)):
            for p in path:
                # Recursively process each element
                self._load_path(p, min_size=min_size, audio_ext=audio_ext)
            return self.path

        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(audio_ext):
                        full_path = os.path.join(root, file)
                        try:
                            if os.path.getsize(full_path) > min_size:
                                self.path.append(full_path)
                        except OSError:
                            continue

        elif os.path.isfile(path):
            low = path.lower()
            if low.endswith(".npy"):
                self.data = np.load(path, allow_pickle=True)
                
                return self.data
            
            elif low.endswith(audio_ext):
                try:
                    if os.path.getsize(path) > min_size:
                        self.path = [path]
                except OSError:
                    self.path = []

        else:
            raise ValueError(f"Invalid path: {path}")

        return self.path


    # def syllable_analysis(self, min_duration=0.1, max_duration=3.0, fscale="mel", noise_profile=[0.0, 0.5],
    #                 n_fft=512, hop_length=128, fps=50, stretch_factor=10.0, display=True):
        
    #     # Set arguments list using path/fmin/fmax from self.data dicts
    #     args_list = [
    #         (
    #         d.get('path'),
    #         1,
    #         15000,
    #         self.audio_filter_chain,
    #         noise_profile,
    #         n_fft, hop_length, fps, min_duration, max_duration, fscale, stretch_factor, display
    #         )
    #         for d in list(self.data) if isinstance(d, dict) and d.get('path') is not None
    #     ]
        
    #     # Detect the region of interests (ROIs)
    #     results = self._multiprocessing(extract_roi_contours, args_list, desc="Detecting syllables of interests (SOIs)")
    #     # Flatten the list of rois
    #     self.syllables = [syl for sublist in results if sublist is not None for syl in sublist]

    #     return self.syllables

        

if __name__ == "__main__":
    # Initialize EcoCASE with a file path or folder path containing audio files
    case = EcoCASE(path='data', multiprocessing=True, rate=48000, fmin=1, fmax=15000)

    # Detect region of interests (ROIs) in the recording
    contours = case.detect_rois(min_duration=0.5, max_duration=10.0, stretch_factor=10.0, display=False)
    # Segment region of interests (ROIs) in the recording
    rois = case.segment_rois(contours, noise_reduction=-20, binary_file_path='rois.npy')
    # Extract embeddings and predict species with BirdNET
    embeddings = case.birdnet_analysis(rois, embeddings=True, predict=True, min_confidence=0.80, binary_file_path='embeddings.npy')

    # # Analyze frequency range of the predicted species
    # results = case.frequency_analysis(species='Chloris chloris_European Greenfinch') 
    # # Analyze repertoire of the species
    # repertoire = case.repertoire_analysis(species='Chloris chloris_European Greenfinch', binary_file_path='chloris_chloris_repertoire.npy') # Chloris chloris_European Greenfinch, Turdus iliacus_Redwing, Alauda arvensis_Eurasian Skylark

    # print(results)

    # case = EcoCASE(data='examples/chloris_chloris_repertoire.npy', multiprocessing=True)
    # results = case.frequency_analysis(species='Chloris chloris_European Greenfingch')
    # case = EcoCASE(path='examples/turdus_iliacus_repertoire.npy', multiprocessing=True)
    # results = case.frequency_analysis(species='Turdus iliacus_Redwing')
    # print(results)

    # Display ecological clusters in an interactive plot
    case.display_clusters(noise_reduction=-20)

    #TODO
    # 1) Add lon and lat to birdnet anaylzer
    # 2) system to choose from the labels species list