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
            x_train, y_train, labels = self._loadPseudoLabeledData(cfg.TRAIN_CACHE_MODE, cfg.TRAIN_CACHE_FILE, on_data_load_end)
        else:
            x_train, y_train, labels = self._loadTrainingData(cfg.TRAIN_CACHE_MODE, cfg.TRAIN_CACHE_FILE, on_data_load_end)
            
        print(f"Done! Loaded {x_train.shape[0]} training samples and {y_train.shape[1]} labels.", flush=True)

        if cfg.AUTOTUNE:
            import gc
            import keras
            import keras_tuner

            # Call callback to initialize progress bar
            if on_trial_result:
                on_trial_result(0)

            class BirdNetTuner(keras_tuner.BayesianOptimization):
                def __init__(self, x_train, y_train, max_trials, executions_per_trial, on_trial_result):
                    super().__init__(
                        max_trials=max_trials,
                        executions_per_trial=executions_per_trial,
                        overwrite=True,
                        directory=autotune_directory,
                        project_name="birdnet_analyzer"
                    )
                    self.x_train = x_train
                    self.y_train = y_train
                    self.on_trial_result = on_trial_result

                def run_trial(self, trial, *args, **kwargs):
                    histories = []
                    hp: keras_tuner.HyperParameters = trial.hyperparameters
                    trial_number = len(self.oracle.trials)

                    for execution in range(int(self.executions_per_trial)):
                        print(f"Running Trial #{trial_number} execution #{execution + 1}", flush=True)

                        # Build model
                        print("Building model...", flush=True)
                        classifier = model.buildLinearClassifier(self.y_train.shape[1], 
                                                                self.x_train.shape[1], 
                                                                hidden_units=hp.Choice("hidden_units", [0, 128, 256, 512, 1024, 2048], default=cfg.TRAIN_HIDDEN_UNITS), 
                                                                dropout=hp.Choice("dropout", [0.0, 0.25, 0.33, 0.5, 0.75, 0.9], default=cfg.TRAIN_DROPOUT))
                        print("Done!", flush=True)

                        # Only allow repeat upsampling in multi-label setting
                        upsampling_choices = ['repeat', 'mean', 'linear'] #SMOTE is too slow

                        if cfg.MULTI_LABEL:
                            upsampling_choices = ['repeat']

                        # Train model
                        print("Training model...", flush=True)
                        classifier, history = model.trainLinearClassifier(
                            classifier,
                            self.x_train,
                            self.y_train,
                            epochs=cfg.TRAIN_EPOCHS,
                            batch_size=hp.Choice("batch_size", [8, 16, 32, 64, 128], default=cfg.TRAIN_BATCH_SIZE),
                            learning_rate=hp.Choice("learning_rate", [0.1, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001], default=cfg.TRAIN_LEARNING_RATE),
                            val_split=cfg.TRAIN_VAL_SPLIT,
                            upsampling_ratio=hp.Choice("upsampling_ratio",[0.0, 0.25, 0.33, 0.5, 0.75, 1.0], default=cfg.UPSAMPLING_RATIO),
                            upsampling_mode=hp.Choice("upsampling_mode", upsampling_choices, default=cfg.UPSAMPLING_MODE), 
                            train_with_mixup=hp.Boolean("mixup", default=cfg.TRAIN_WITH_MIXUP),
                            train_with_label_smoothing=hp.Boolean("label_smoothing", default=cfg.TRAIN_WITH_LABEL_SMOOTHING),
                        )

                        # Get the best validation loss
                        # Is it maybe better to return the negative val_auprc??
                        best_val_loss = history.history["val_loss"][np.argmin(history.history["val_loss"])]
                        histories.append(best_val_loss)

                        print(f"Finished Trial #{trial_number} execution #{execution + 1}. best validation loss: {best_val_loss}", flush=True)

                    keras.backend.clear_session()
                    del classifier
                    del history
                    gc.collect()

                    # Call the on_trial_result callback
                    if self.on_trial_result:
                        self.on_trial_result(trial_number)

                    return histories

            tuner = BirdNetTuner(x_train=x_train, y_train=y_train, max_trials=cfg.AUTOTUNE_TRIALS, executions_per_trial=cfg.AUTOTUNE_EXECUTIONS_PER_TRIAL, on_trial_result=on_trial_result)
            tuner.search()
            best_params = tuner.get_best_hyperparameters()[0]
            print("Best params: ")
            print("hidden_units: ", best_params["hidden_units"])
            print("dropout: ", best_params["dropout"])
            print("batch_size: ", best_params["batch_size"])
            print("learning_rate: ", best_params["learning_rate"])
            print("upsampling_mode: ", best_params["upsampling_mode"])
            print("upsampling_ratio: ", best_params["upsampling_ratio"])
            print("mixup: ", best_params["mixup"])
            print("label_smoothing: ", best_params["label_smoothing"])
            cfg.TRAIN_HIDDEN_UNITS = best_params["hidden_units"]
            cfg.TRAIN_DROPOUT = best_params["dropout"]
            cfg.TRAIN_BATCH_SIZE = best_params["batch_size"]
            cfg.TRAIN_LEARNING_RATE = best_params["learning_rate"]
            cfg.UPSAMPLING_MODE = best_params["upsampling_mode"]
            cfg.UPSAMPLING_RATIO = best_params["upsampling_ratio"]
            cfg.TRAIN_WITH_MIXUP = best_params["mixup"]
            cfg.TRAIN_WITH_LABEL_SMOOTHING = best_params["label_smoothing"]
            

        # Build model
        print("Building model...", flush=True)
        classifier = model.buildLinearClassifier(y_train.shape[1], x_train.shape[1], cfg.TRAIN_HIDDEN_UNITS, cfg.TRAIN_DROPOUT)
        print("Done!", flush=True)
        
        # Train model
        print("Training model...", flush=True)
        classifier, history = model.trainLinearClassifier(
            classifier,
            x_train,
            y_train,
            epochs=cfg.TRAIN_EPOCHS,
            batch_size=cfg.TRAIN_BATCH_SIZE,
            learning_rate=cfg.TRAIN_LEARNING_RATE,
            val_split=cfg.TRAIN_VAL_SPLIT,
            upsampling_ratio=cfg.UPSAMPLING_RATIO,
            upsampling_mode=cfg.UPSAMPLING_MODE,
            train_with_mixup=cfg.TRAIN_WITH_MIXUP,
            train_with_label_smoothing=cfg.TRAIN_WITH_LABEL_SMOOTHING,
            on_epoch_end=on_epoch_end,
        )

        # Best validation AUPRC (at minimum validation loss)
        best_val_auprc = history.history["val_AUPRC"][np.argmin(history.history["val_loss"])]
        best_val_auroc = history.history["val_AUROC"][np.argmin(history.history["val_loss"])]

        if cfg.TRAINED_MODEL_OUTPUT_FORMAT == "both":
            model.save_raven_model(classifier, custom_classifier_path, labels)
            model.saveLinearClassifier(classifier, custom_classifier_path, labels, mode=cfg.TRAINED_MODEL_SAVE_MODE)
        elif cfg.TRAINED_MODEL_OUTPUT_FORMAT == "tflite":
            model.saveLinearClassifier(classifier, custom_classifier_path, labels, mode=cfg.TRAINED_MODEL_SAVE_MODE)
        elif cfg.TRAINED_MODEL_OUTPUT_FORMAT == "raven":
            model.save_raven_model(classifier, custom_classifier_path, labels)
        else:
            raise ValueError(f"Unknown model output format: {cfg.TRAINED_MODEL_OUTPUT_FORMAT}")

        print(f"Done! Best AUPRC: {best_val_auprc}, Best AUROC: {best_val_auroc}", flush=True)
        print(f"Path to trained classifier model output: {custom_classifier_path}")

        return history