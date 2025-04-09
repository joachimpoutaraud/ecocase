import os
from functools import partial
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

import audio
import analyze
import config as cfg
import model
import utils


class BirdNET():
    
    def __init__(
                self,
                verbose = True,
                save_path = None,
                download = True,
                ):

        self.verbose = verbose
        self.save_path = save_path
        self.download = download

    def _loadAudioFile(self, f, label_vector, config, offset=None, duration=None):
        """Load an audio file and extract features.
        Args:
            f: Path to the audio file.
            label_vector: The label vector for the file.
        Returns:
            A tuple of (x_train, y_train).
        """    

        x_train = []
        y_train = []

        # restore config in case we're on Windows to be thread save
        cfg.setConfig(config)

        # Try to load the audio file
        try:
            # Load audio
            if offset is not None and duration is not None:
                sig, rate = audio.openAudioFile(f, offset=offset, duration=duration, fmin=cfg.BANDPASS_FMIN, fmax=cfg.BANDPASS_FMAX)
            else:
                sig, rate = audio.openAudioFile(f, duration=cfg.SIG_LENGTH if cfg.SAMPLE_CROP_MODE == "first" else None, fmin=cfg.BANDPASS_FMIN, fmax=cfg.BANDPASS_FMAX)

        # if anything happens print the error and ignore the file
        except Exception as e:
            # Print Error
            print(f"\t Error when loading file {f}", flush=True)
            print(f"\t {e}", flush=True)
            return np.array([]), np.array([])

        # Crop training samples
        if cfg.SAMPLE_CROP_MODE == "center":
            sig_splits = [audio.cropCenter(sig, rate, cfg.SIG_LENGTH)]
        elif cfg.SAMPLE_CROP_MODE == "first":
            sig_splits = [audio.splitSignal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)[0]]
        else:
            sig_splits = audio.splitSignal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)

        # Get feature embeddings
        batch_size = 1 # turns out that batch size 1 is the fastest, probably because of having to resize the model input when the number of samples in a batch changes
        for i in range(0, len(sig_splits), batch_size):
            batch_sig = sig_splits[i:i+batch_size]
            batch_label = [label_vector] * len(batch_sig)
            embeddings = model.embeddings(batch_sig)

            # Add to training data
            x_train.extend(embeddings)
            y_train.extend(batch_label)

        return x_train, y_train


    def _extractFeatures(self, row):
        
        """
        Extracts features from the chunks of an audio file from the row of a dataframe.

        Args:
            row (pd.Series): Row of a pandas Dataframe.
            
        Returns:
            pd.DataFrame: DataFrame containing all embeddings extracted from the audio chunks.
        """

        fpath = row.file
        fileLengthSeconds = row.length 
        
        # Check for potential audio offset
        if 'offset' in row:
            offset = row.offset
        else:
            offset = 0
        
        # Check whether to use custom classifier or default config classifier
        if self.custom_classifier_path is not None:
            filename, _ = os.path.splitext(self.custom_classifier_path)
            # Update model and label file paths
            cfg.MODEL_PATH = self.custom_classifier_path
            cfg.LABELS_FILE = f"{filename}_Labels.txt"

        # Initialize dataframe for processing row    
        df = pd.DataFrame([row])

        # Process each chunk
        try:
            while offset < fileLengthSeconds:
                chunks = analyze.getRawAudioFromFile(fpath, offset, cfg.FILE_SPLITTING_DURATION)
                start, end = offset, cfg.SIG_LENGTH + offset
                samples = []
                timestamps = []
                embeddings = []

                for c in range(len(chunks)):
                    # Add to batch
                    samples.append(chunks[c])
                    timestamps.append([start, end])

                    # Advance start and end
                    start += cfg.SIG_LENGTH - cfg.SIG_OVERLAP
                    end = start + cfg.SIG_LENGTH

                    # Check if batch is full or last chunk
                    if len(samples) < cfg.BATCH_SIZE and c < len(chunks) - 1:
                        continue

                    # Prepare sample and pass through model
                    data = np.array(samples, dtype="float32")
                    e = model.embeddings(data)
                    # Append embeddings to list
                    embeddings.append(e[0])

                    # Reset batch
                    samples = []
                    
                # Copy audio chunk rows into dataframe
                df = pd.DataFrame([df.iloc[0].copy() for _ in range(1, len(embeddings) + 1)])
                data = np.concatenate([np.asarray(timestamps), np.asarray(embeddings)], axis=1)
                
                # Format timestamps and embeddings into new dataframe 
                cols = ['start', 'end'] 
                cols += ['emb' + str(idx).zfill(4) for idx in range(1, len(embeddings[0]) + 1)] 
                new_df = pd.DataFrame(data=data, columns=cols)
                
                # Concatenate dataFrames
                df = pd.concat([df.reset_index(drop=True), new_df.reset_index(drop=True)], axis=1)

                offset = offset + cfg.FILE_SPLITTING_DURATION

        except Exception as ex:
            # Write error log
            print(f"Error: Cannot analyze audio file {fpath}.", flush=True)
            utils.writeErrorLog(ex)
            return
        
        try:
            # Save embeddings
            return df

        except Exception as ex:
            # Write error log
            print(f"Error: Cannot save embeddings for {fpath}.", flush=True)
            utils.writeErrorLog(ex)
            return
    
        
    def _loadTrainingData(self, cache_mode="none", cache_file="", progress_callback=None):
        """Loads the data for training.

        Reads all subdirectories of "config.TRAIN_DATA_PATH" and uses their names as new labels.

        These directories should contain all the training data for each label.

        If a cache file is provided, the training data is loaded from there.

        Args:
            cache_mode: Cache mode. Can be 'none', 'load' or 'save'. Defaults to 'none'.
            cache_file: Path to cache file.

        Returns:
            A tuple of (x_train, y_train, labels).
        """
        # Load from cache
        if cache_mode == "load":
            if os.path.isfile(cache_file):
                print(f"\t...loading from cache: {cache_file}", flush=True)
                x_train, y_train, labels, cfg.BINARY_CLASSIFICATION, cfg.MULTI_LABEL = utils.loadFromCache(cache_file)
                return x_train, y_train, labels
            else:
                print(f"\t...cache file not found: {cache_file}", flush=True)

        # Get list of subfolders as labels
        folders = list(sorted(utils.list_subdirectories(cfg.TRAIN_DATA_PATH)))

        # Read all individual labels from the folder names
        labels = []

        for folder in folders:
            labels_in_folder = folder.split(',')
            for label in labels_in_folder:
                if not label in labels:
                    labels.append(label)

        # Sort labels
        labels = list(sorted(labels))

        # Get valid labels
        valid_labels = [l for l in labels if not l.lower() in cfg.NON_EVENT_CLASSES and not l.startswith("-")] 

        # Check if binary classification
        cfg.BINARY_CLASSIFICATION = len(valid_labels) == 1

        # Validate the classes for binary classification
        if cfg.BINARY_CLASSIFICATION:
            if len([l for l in folders if l.startswith("-")]) > 0:
                raise Exception("Negative labels can't be used with binary classification", "validation-no-negative-samples-in-binary-classification")
            if len([l for l in folders if l.lower() in cfg.NON_EVENT_CLASSES]) == 0:
                raise Exception("Non-event samples are required for binary classification", "validation-non-event-samples-required-in-binary-classification")

        # Check if multi label
        cfg.MULTI_LABEL = len(valid_labels) > 1 and any(',' in f for f in folders)

        # Check if multi-label and binary classficication 
        if cfg.BINARY_CLASSIFICATION and cfg.MULTI_LABEL:
            raise Exception("Error: Binary classfication and multi-label not possible at the same time")

        # Only allow repeat upsampling for multi-label setting
        if cfg.MULTI_LABEL and cfg.UPSAMPLING_RATIO > 0 and cfg.UPSAMPLING_MODE != 'repeat':
            raise Exception("Only repeat-upsampling ist available for multi-label", "validation-only-repeat-upsampling-for-multi-label")

        # Load training data
        x_train = []
        y_train = []

        for folder in folders:

            # Get label vector
            label_vector = np.zeros((len(valid_labels),), dtype="float32")

            folder_labels = folder.split(',')

            for label in folder_labels:
                if not label.lower() in cfg.NON_EVENT_CLASSES and not label.startswith("-"):
                    label_vector[valid_labels.index(label)] = 1
                elif label.startswith("-") and label[1:] in valid_labels: # Negative labels need to be contained in the valid labels
                    label_vector[valid_labels.index(label[1:])] = -1

            # Get list of files
            # Filter files that start with '.' because macOS seems to them for temp files.
            files = filter(
                os.path.isfile,
                (
                    os.path.join(cfg.TRAIN_DATA_PATH, folder, f)
                    for f in sorted(os.listdir(os.path.join(cfg.TRAIN_DATA_PATH, folder)))
                    if not f.startswith(".") and f.rsplit(".", 1)[-1].lower() in cfg.ALLOWED_FILETYPES
                ),
            )

            # Load files using thread pool       
            with Pool(cfg.CPU_THREADS) as p:
                tasks = []
                for f in files:
                    task = p.apply_async(partial(self._loadAudioFile, f=f, label_vector=label_vector, config=cfg.getConfig()))
                    tasks.append(task)

                # Wait for tasks to complete and monitor progress with tqdm
                num_files_processed = 0
                with tqdm(total=len(tasks), desc=f"Loading '{folder}'", unit='f') as progress_bar:
                    for task in tasks:
                        result = task.get()
                        # Make sure result is not empty 
                        # Empty results might be caused by errors when loading the audio file
                        # TODO: We should check for embeddings size in result, otherwise we can't add them to the training data
                        if len(result[0]) > 0:                        
                            x_train += result[0]
                            y_train += result[1]
                        num_files_processed += 1
                        progress_bar.update(1)
                        if progress_callback:
                            progress_callback(num_files_processed, len(tasks), folder)
        
        # Convert to numpy arrays
        x_train = np.array(x_train, dtype="float32")
        y_train = np.array(y_train, dtype="float32")        
        
        # Save to cache?
        if cache_mode == "save":
            print(f"\t...saving training data to cache: {cache_file}", flush=True)
            try:
                # Only save the valid labels
                utils.saveToCache(cache_file, x_train, y_train, valid_labels)
            except Exception as e:
                print(f"\t...error saving cache: {e}", flush=True)

        # Return only the valid labels for further use
        return x_train, y_train, valid_labels


    def _loadPseudoLabeledData(self, cache_mode="none", cache_file="", progress_callback=None):
        """
        Loads the pseudo-labeled dataset for training.

        Reads all categories in the dataframe and uses their names as new labels.
        The dataframe should contain a column 'file' with all the training files path or urls for each label.

        Args:
            cache_mode: Cache mode. Can be 'none', 'load' or 'save'. Defaults to 'none'.
            cache_file: Path to cache file.

        Returns:
            A tuple of (x_train, y_train, labels).
        """

        # Load from cache
        if cache_mode == "load":
            if os.path.isfile(cache_file):
                print(f"\t...loading from cache: {cache_file}", flush=True)
                x_train, y_train, labels, cfg.BINARY_CLASSIFICATION, cfg.MULTI_LABEL = utils.loadFromCache(cache_file)
                return x_train, y_train, labels
            else:
                print(f"\t...cache file not found: {cache_file}", flush=True)

        # Get the unique categories
        categories = self.df.categories.unique()
        
        # Get the embeddings
        x_train = self.df.loc[:, self.df.columns.str.startswith("emb")]
        
        # Perform one-hot encoding on categories
        one_hot_encoded = pd.get_dummies(self.df.categories) 
        y_train = one_hot_encoded.values
        
        # # Initialize the LabelEncoder
        # from sklearn.preprocessing import LabelEncoder 
        # label_encoder = LabelEncoder() 
        # # Fit and transform the column 
        # y_train = label_encoder.fit_transform(self.df['categories'])
        
        # Convert to NumPy array         
        x_train = np.array(x_train, dtype="float32")
        y_train = np.array(y_train, dtype="float32")
        
        # Save to cache?
        if cache_mode == "save":
            print(f"\t...saving training data to cache: {cache_file}", flush=True)
            try:
                # Only save the valid labels
                utils.saveToCache(cache_file, x_train, y_train, categories)
            except Exception as e:
                print(f"\t...error saving cache: {e}", flush=True)

        # Return only the valid labels for further use
        return x_train, y_train, categories


    def trainModel(self, on_epoch_end=None, on_trial_result=None, on_data_load_end=None, autotune_directory="autotune", custom_classifier_path=cfg.CUSTOM_CLASSIFIER):
        """Trains a custom classifier.

        Args:
            on_epoch_end: A callback function that takes two arguments `epoch`, `logs`.
            custom_classifier_path (str, optional): Path to the output custom classifier. Default is cfg.CUSTOM_CLASSIFIER.

        Returns:
            A keras `History` object, whose `history` property contains all the metrics.
        """

        # Load training data
        print("Loading training data...", flush=True)
        
        if cfg.TRAIN_DATA_PATH is None:    
            x_train, y_train, categories = self._loadPseudoLabeledData(cfg.TRAIN_CACHE_MODE, cfg.TRAIN_CACHE_FILE, on_data_load_end)
        else:
            x_train, y_train, categories = self._loadTrainingData(cfg.TRAIN_CACHE_MODE, cfg.TRAIN_CACHE_FILE, on_data_load_end)
            
        print(f"Done! Loaded {x_train.shape[0]} training samples and {y_train.shape[1]} labels.\n", flush=True)
              

        from umap.parametric_umap import ParametricUMAP       
        from tensorflow import keras

        # Build the ConvNet encoder        
        encoder = keras.Sequential([
            keras.layers.Input(shape=(x_train.shape[1],)),
            keras.layers.Flatten(),
            keras.layers.Dense(units=512, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units=256, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units=128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units=64, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units=32, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units=16, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units=8, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units=4, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(units=2, name="z")
        ])

        keras_fit_kwargs = {"callbacks": [
            keras.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=10**-2,
                patience=10,
                verbose=1,
            )
        ]}

        embedder = ParametricUMAP(
                                  encoder=encoder, 
                                  batch_size=cfg.TRAIN_BATCH_SIZE, 
                                  random_state=cfg.RANDOM_SEED,
                                  metric=cfg.METRIC,
                                  keras_fit_kwargs=keras_fit_kwargs,
                                  verbose=self.verbose,
                                  )
        
        embedder.n_training_epochs = cfg.TRAIN_EPOCHS

        # Fit ParametricUMAP model with pseudo-labeled data
        pseudo_labels = np.argmax(y_train, axis=1)
        embedder.fit(x_train, y=pseudo_labels)

        # # Save ParametricUMAP model
        # embedder.save(custom_classifier_path)
        
        return embedder