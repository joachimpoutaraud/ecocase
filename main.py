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
    def __init__(self, path, rate=48000, fmin=1, fmax=15000, lat=None, lon=None, week_48=-1, min_confidence=0.25, audio_filter_chain=None, multiprocessing=True, verbose=True, random_state=2025):
        
        self.rate = rate
        self.fmin = fmin
        self.fmax = fmax
        self.lat = lat
        self.lon = lon
        self.week_48 = week_48
        self.min_confidence = min_confidence
        self.multiprocessing = multiprocessing
        self.verbose = verbose
        self.random_state = random_state

        # Set random seed for reproducibility    
        np.random.seed(self.random_state)

        # Load path(s) or .npy data
        self._load_path(path)
        
        if self.lon is not None and self.lat is not None:
            # Get species list based on location
            self.species_list = self._species_list(self.lon, self.lat, self.week_48)
        else:
            self.species_list = []
        
        # Get the audio filter chain
        self.audio_filter_chain = audio_filter_chain or ",".join([
            # Resample audio file
            f"aresample=resampler=soxr:out_sample_rate={rate}",
            # Convert to mono
            f"aformat=channel_layouts=mono",
        ])
    

    def detect_rois(self, min_duration=0.5, max_duration=10.0, fscale="mel", noise_floor="auto", noise_profile=[0.0, 0.5],
                    n_fft=1024, pps=256, display=False):
        """
        Detect region of interests (ROIs) from audio files.
        
        Parameters:
            min_duration (float): Minimum duration of ROIs in seconds.
            max_duration (float): Maximum duration of ROIs in seconds.
            fscale (str): Frequency scale for spectrogram ('mel', 'log', 'linear').
            noise_floor (str): Set the noise floor in dB, allowed range is -80 to -20. Default is 'auto' which enables automatic noise floor tracking.
            noise_profile (list): Noise profile for denoising audio file with FFT [min, max] (seconds).
            n_fft (int): Number of FFT components for spectrogram. Default value is 1024.
            pps (int): Number of pixels per second (pps) for spectrogram. Allowed range is from 1 to 1024. Default value is 256.
            display (bool): Whether to display detected ROIs in a plot. Default is False.
            
        Returns:
            rois (list): List of detected ROIs dictionaries.
        """
        
        # Set arguments list
        args_list = [(path, self.fmin, self.fmax, self.audio_filter_chain, noise_floor, noise_profile,
                    n_fft, pps, min_duration, max_duration, fscale, display)
                    for path in self.path]
        
        # Detect the region of interests (ROIs)
        results = self._multiprocessing(extract_roi_contours, args_list, desc="Detecting region of interests (ROIs)")
        # Flatten the list of rois
        rois = [roi for sublist in results if sublist is not None for roi in sublist]

        return rois
    

    def segment_rois(self, rois, download=False, save_binary=None):
        """
        Segment region of interests (ROIs) from audio files and optionally save results as a .npy file.
        
        Parameters:
            rois (list): List of ROIs dictionaries.
            download (bool): Whether to save the segmented ROIs as audio files.
            save_binary (str): Path to save the .npy results file. Default is None (no saving).
            
        Returns:
            rois (list): List of segmented ROIs data.
        """

        # Set arguments list
        args_list = [(roi['path'], roi['tmin'], roi['tmax'], roi['fmin'], roi['fmax'], self.audio_filter_chain, download, self.rate) for roi in rois]

        # Segment the region of interests (ROIs)
        self.rois = self._multiprocessing(extract_roi_audio, args_list, desc="Segmenting region of interests (ROIs)")

        # Save rois to a binary file in NumPy .npy format
        if save_binary is not None:
            np.save(save_binary, self.rois)    

        return self.rois  


    def birdnet_analysis(self, rois, embeddings=False, predict=True, keep_best=True, save_binary=None):
        """
        Analyze region of interests (ROIs) using BirdNET and optionally save results as a .npy file.

        Parameters:
            rois (list): List of ROIs dictionaries.
            embeddings (bool): Whether to extract BirdNET embeddings or not.
            predict (bool): Whether to run BirdNET prediction or not.
            keep_best (bool): Whether to keep only one embedding with the best confidence score for ROI longer than 3s.
            save_binary (str): Path to save the .npy results file. Default is None (no saving).

        Returns:
            data (list): Analysis results.
        """

        # Check if rois is a binary file path
        if isinstance(rois, str):
            rois = np.load(rois, allow_pickle=True)

        # Set arguments list
        args_list = [roi for roi in rois]
        # Segment the region of interests (ROIs)
        results = self._multiprocessing(prepare_segments, args_list, desc="Preparing segments for BirdNET analysis")
        # Flatten the list of segments
        segments = [segment for sublist in results if sublist is not None for segment in sublist]  
        # Set arguments list
        args_list = [(seg, embeddings, predict, self.species_list) for seg in segments]
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
        if save_binary is not None:
            np.save(save_binary, self.data)

        return self.data


    # def train(self, species=None, latent_dim=128, epochs=100, dropout_rate=0.3, batch_size=64, save_binary=None):
    #     import random
    #     # Seed Python and NumPy RNGs before importing TensorFlow
    #     random.seed(self.random_state)
    #     np.random.seed(self.random_state)

        # import tensorflow as tf
        # from tensorflow import keras
        # from tensorflow.keras.optimizers import Adam
        # from tensorflow.keras.losses import CosineSimilarity
        
        # if species is not None:
        #     # Filter embeddings with the selected species
        #     self.data = [d for d in self.data if isinstance(d, dict) and d.get('prediction', '').lower() == species.lower()]

        # # Get the embeddings
        # X = np.array([d['embeddings'] for d in self.data if d.get('embeddings') is not None]).astype(np.float32)

        # # Train an autoencoder to reduce to X dimensions
        # tf.random.set_seed(self.random_state)
        # keras_verbose = 1 if self.verbose else 0

        # input_dim = X.shape[1]

        # # Build a simple fully-connected autoencoder with a bottleneck of dimension `latent_dim`
        # inputs = keras.Input(shape=(input_dim,))
        # units1, units2 = 512, 256

        # x = keras.layers.Dense(units1, use_bias=False)(inputs)
        # x = keras.layers.Activation('relu')(x)
        # x = keras.layers.LayerNormalization()(x)
        # x = keras.layers.Dropout(dropout_rate)(x)

        # x = keras.layers.Dense(units2, use_bias=False)(x)
        # x = keras.layers.Activation('relu')(x)
        # x = keras.layers.LayerNormalization()(x) 
        # x = keras.layers.Dropout(dropout_rate)(x)

        # latent = keras.layers.Dense(latent_dim, activation='linear', name='bottleneck')(x)
        # latent = keras.layers.LayerNormalization()(latent)

        # x = keras.layers.Dense(units2, use_bias=False)(latent)
        # x = keras.layers.Activation('relu')(x)
        # x = keras.layers.LayerNormalization()(x)
        # x = keras.layers.Dropout(dropout_rate)(x)

        # x = keras.layers.Dense(units1, use_bias=False)(x)
        # x = keras.layers.Activation('relu')(x)
        # x = keras.layers.LayerNormalization()(x)
        # x = keras.layers.Dropout(dropout_rate)(x)

        # outputs = keras.layers.Dense(input_dim, activation='linear')(x)
        # autoencoder = keras.Model(inputs, outputs)
        # encoder = keras.Model(inputs, autoencoder.get_layer('bottleneck').output)

        # # Configure the model for training
        # autoencoder.compile(optimizer=Adam(), loss=CosineSimilarity())

        # # Train with early stopping
        # es = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=keras_verbose)

        # autoencoder.fit(
        #                 X, X,
        #                 epochs=epochs,
        #                 batch_size=batch_size,
        #                 validation_split=0.2 if X.shape[0] > 10 else 0.0,
        #                 callbacks=[es],
        #                 verbose=keras_verbose,
        #                 shuffle=False
        #                 )

        # # Extract reduced X-dim embeddings from encoder bottleneck
        # embeddings = encoder.predict(X)

        # # Update embeddings
        # for i, d in enumerate(self.data):
        #     d['embeddings'] = embeddings[i]

        # # Save data to a binary file in NumPy .npy format
        # if save_binary is not None:
        #     np.save(save_binary, self.data)

        # return self.data


    def frequency_analysis(self, species=None):
        """
        Analyze frequency range of detected species in the ROIs.

        Parameters:
            species (str): Specific species to analyze. If None, analyze all detected species.

        Returns:
            results_dict (dict): Dictionary with frequency analysis results.
        """

        # Filter out low-confidence predictions by marking them as "Unknown"
        if self.min_confidence is not None:
            for item in self.data:
                conf = item.get('confidence')
                if conf < float(self.min_confidence):
                    if 'repertoire' in self.data[0]:
                        item['repertoire'] = 'Unknown'
                    else:
                        item['prediction'] = 'Unknown'
                    item['confidence'] = 0.0

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


    def repertoire_analysis(self, species, min_cluster_size=15, metric='cosine', keep_noise=False, save_binary=None):
        import umap
        import hdbscan
        from sklearn.metrics import pairwise_distances

        # Filter out low-confidence predictions by marking them as "Unknown"
        if self.min_confidence is not None:
            for item in self.data:
                conf = item.get('confidence')
                if conf < float(self.min_confidence):
                    if 'repertoire' in self.data[0]:
                        item['repertoire'] = 'Unknown'
                    else:
                        item['prediction'] = 'Unknown'
                    item['confidence'] = 0.0
        


        # import random
        # # Seed Python and NumPy RNGs before importing TensorFlow
        # random.seed(self.random_state)
        # np.random.seed(self.random_state)

        # import tensorflow as tf
        # from tensorflow import keras
        # from tensorflow.keras.optimizers import Adam
        # from tensorflow.keras.losses import CosineSimilarity

        # # Filter embeddings with the selected species
        # self.data = [d for d in self.data if isinstance(d, dict) and d.get('prediction', '').lower() == species.lower()]
        # # Get the embeddings
        # X = np.array([d['embeddings'] for d in self.data if d.get('embeddings') is not None]).astype(np.float32)

        # # Train an autoencoder to reduce to X dimensions
        # tf.random.set_seed(self.random_state)
        # keras_verbose = 1 if self.verbose else 0

        # input_dim = X.shape[1]

        # # Build a simple fully-connected autoencoder with a bottleneck of dimension `latent_dim`
        # inputs = keras.Input(shape=(input_dim,))
        # units1, units2 = 128, 64
        # latent_dim = 32

        # dropout_rate = 0.3
        # epochs = 100
        # batch_size = 64

        # x = keras.layers.Dense(units1, use_bias=False)(inputs)
        # x = keras.layers.Activation('relu')(x)
        # x = keras.layers.LayerNormalization()(x)
        # x = keras.layers.Dropout(dropout_rate)(x)

        # x = keras.layers.Dense(units2, use_bias=False)(x)
        # x = keras.layers.Activation('relu')(x)
        # x = keras.layers.LayerNormalization()(x) 
        # x = keras.layers.Dropout(dropout_rate)(x)

        # latent = keras.layers.Dense(latent_dim, activation='linear', name='bottleneck', kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
        # latent = keras.layers.LayerNormalization()(latent)
        # latent = keras.layers.Dropout(dropout_rate)(latent)

        # x = keras.layers.Dense(units2, use_bias=False)(latent)
        # x = keras.layers.Activation('relu')(x)
        # x = keras.layers.LayerNormalization()(x)
        # x = keras.layers.Dropout(dropout_rate)(x)

        # x = keras.layers.Dense(units1, use_bias=False)(x)
        # x = keras.layers.Activation('relu')(x)
        # x = keras.layers.LayerNormalization()(x)
        # x = keras.layers.Dropout(dropout_rate)(x)

        # outputs = keras.layers.Dense(input_dim, activation='linear')(x)
        # autoencoder = keras.Model(inputs, outputs)
        # encoder = keras.Model(inputs, autoencoder.get_layer('bottleneck').output)

        # # Configure the model for training
        # autoencoder.compile(optimizer=Adam(), loss=CosineSimilarity())

        # # Train with early stopping
        # es = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=keras_verbose)

        # autoencoder.fit(
        #                 X, X,
        #                 epochs=epochs,
        #                 batch_size=batch_size,
        #                 validation_split=0.2 if X.shape[0] > 10 else 0.0,
        #                 callbacks=[es],
        #                 verbose=keras_verbose,
        #                 shuffle=False
        #                 )

        # # Extract reduced X-dim embeddings from encoder bottleneck
        # emb = encoder.predict(X)



        # Filter embeddings with the selected species
        self.data = [d for d in self.data if isinstance(d, dict) and d.get('prediction', '').lower() == species.lower()]
        # self.data = [d for d in self.data if isinstance(d, dict) and d.get('repertoire', '').lower() == 'Cluster 1'.lower()]

        # Get the embeddings
        X = np.array([d['embeddings'] for d in self.data if d.get('embeddings') is not None])

        best_score = -np.inf
        best_labels = None
        best_params = None
        best_embeddings = None

        for c in [15]:
            # UMAP dimensionality reduction
            reducer = umap.UMAP(n_neighbors=min_cluster_size, n_components=100, metric=metric, random_state=self.random_state, verbose=self.verbose)
            emb = reducer.fit_transform(X)

            # HDBSCAN clustering with DBCV score evaluation
            precomputed_distances = pairwise_distances(emb, metric=metric).astype('float64')
            model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed').fit(precomputed_distances)
            score = hdbscan.validity.validity_index(precomputed_distances, model.labels_, metric='precomputed', d=emb.shape[1])

            if np.isnan(score):
                score = -np.inf

            if score > best_score:
                best_score = score
                best_labels = model.labels_
                best_embeddings = emb
                best_params = {'min_cluster_size': min_cluster_size, 'metric': metric}

        # expose chosen results to the rest of the method
        embeddings = best_embeddings
        best_score = float(best_score)

        # Update embeddings and add 'repertoire' key with label values
        for i, d in enumerate(self.data):
            # store the reduced vector (UMAP) rather than a row of the distance matrix
            d['embeddings'] = embeddings[i]
            if i < len(best_labels):
                label = best_labels[i]
                d['repertoire'] = f"Cluster {label}" if label != -1 else "Noise"

        if self.verbose:
            n_clusters = len(np.unique(best_labels[best_labels >= 0]))
            print(f'Number of clusters found (excluding noise): {n_clusters} {best_params}')
            print(f"DBCV score :{best_score}")

        if keep_noise is not True:
            # Remove all entries labeled as "Noise"
            self.data = [d for d in self.data if d.get('repertoire') != "Noise"]
            
        # Save data to a binary file in NumPy .npy format
        if save_binary is not None:
            np.save(save_binary, self.data)

        return self.data
    
    
    def display_clusters(self, frame_width=1400, frame_height=600, point_size=5, port=8080):
        # Display clusters in an interactive plot with Bokeh server
        run_bokeh_server(self.data, self.audio_filter_chain, self.min_confidence, frame_width, frame_height, point_size, port, self.random_state)


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
    
    
    def _species_list(self, lon, lat, week_48):
        """
        Retrieve species list based on geographic location and week.

        Parameters:
            lon (float): Longitude.
            lat (float): Latitude.
            week_48 (bool): Whether to use week 48 species list for northern hemisphere.
        Returns:
            list: List of species names.    
        """
        from birdnetlib.species import SpeciesList

        self.species_class = SpeciesList()
        return self.species_class.return_list_for_analyzer(lat=lat, lon=lon, week_48=week_48, threshold=0.03)
            
    
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
        

if __name__ == "__main__":
    # Initialize EcoCASE with a file path or folder path containing audio files
    case = EcoCASE(path='out.wav', multiprocessing=True, rate=48000, fmin=1, fmax=15000, min_confidence=0.25, lat=4.937, lon=-52.326)

    # Detect region of interests (ROIs) in the recording
    contours = case.detect_rois(min_duration=0.2, max_duration=10.0, display=True)
    # Segment region of interests (ROIs) in the recording
    rois = case.segment_rois(contours)#, save_binary='rois.npy')
    # Extract embeddings and predict species with BirdNET
    data = case.birdnet_analysis(rois, embeddings=True, predict=True, save_binary='data.npy')   

    # Analyze frequency range of the predicted species
    results = case.frequency_analysis()
    # # Analyze repertoire of the species
    # repertoire = case.repertoire_analysis(species='Turdus iliacus_Redwing')#, save_binary='repertoire.npy') # Chloris chloris_European Greenfinch, Turdus iliacus_Redwing, Alauda arvensis_Eurasian Skylark
    
    # Display ecological clusters in an interactive plot
    case.display_clusters(port=8080)

    #TODO
    # 1) check best DBCV score
    # 2) system to choose from the labels species list
