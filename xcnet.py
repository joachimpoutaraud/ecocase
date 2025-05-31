import os
import shutil
import pandas as pd
import numpy as np

import tqdm
from pathlib import Path
from functools import partial
from multiprocessing.pool import Pool

from birdnet_analyzer import config as cfg
from birdnet_analyzer import audio
from birdnet_analyzer import model

from dataset import XenoCanto
from clustering import cluster


class XCNet(XenoCanto):

    def __init__(
                self,
                params_xc = cfg.PARAMS_XC,
                species_xc = cfg.SPECIES_XC,
                num_files = cfg.NUM_FILES,
                format_time = False,
                format_date = False,
                verbose = True,
                save_path = None,
                overwrite = False,
                ):

        self.params_xc = params_xc
        self.species_xc = species_xc
        self.num_files = num_files
        self.format_time = format_time
        self.format_date = format_date 
        self.verbose = verbose
        self.save_path = save_path
        self.overwrite = overwrite

        # Check whether root directory is existing or not
        if self.save_path is not None:   
            if os.path.exists(self.save_path):
                if self.overwrite:
                    # Remove directory
                    shutil.rmtree(str(self.save_path))
                    # Save dataframe
                    os.makedirs(self.save_path)      
            else:      
                os.makedirs(self.save_path)  


    def train(self, custom_classifier=cfg.CUSTOM_CLASSIFIER, cache_mode="load", cache_file="", on_epoch_end=None, on_trial_result=None, on_data_load_end=None, autotune_directory="autotune"):

        from birdnet_analyzer.train import utils
        
        config = cfg.get_config()

        # Set the configuration parameters
        if self.save_path is not None:
            # Format save_path into Path
            save_path = Path(self.save_path)
            custom_classifier = str(save_path / custom_classifier)

            # Save and append directory
            config['CUSTOM_CLASSIFIER'] = custom_classifier
            config['TRAIN_CACHE_FILE'] = str(save_path / cache_file)
        else:
            config['CUSTOM_CLASSIFIER'] = custom_classifier
            config['TRAIN_CACHE_FILE'] = cache_file

        config['TRAIN_CACHE_MODE'] = cache_mode

        cfg.set_config(config)

        # Train the model
        utils.train_model(on_epoch_end=on_epoch_end, on_trial_result=on_trial_result, on_data_load_end=on_data_load_end, autotune_directory=autotune_directory)

        if self.verbose:
            print(f"\nSaving custom classifier: {custom_classifier}", flush=True)


    def labeling(self, dataset=None, custom_classifier=cfg.CUSTOM_CLASSIFIER, csv_filename=None, cache_filename=None):
        """ 
        Data-labeling using clustering or classification. By default, it uses clustering to label the data.
        If a custom classifier is provided, it will use the classifier to predict the labels of the data.

        Args:   
            dataset (str or pd.DataFrame, optional): Path to the dataset. It can be a CSV file or a folder path containing audio files. Default is None.
                If None, the dataset is queried from Xeno-Canto according to the Xeno-Canto parameters in the configuration file.
            custom_classifier (str, optional): Path to the custom classifier file. Default is None.
            csv_filename (str, optional): Name of the CSV file to save the labeled data. Default is None.
                If None, the labeled data will not be saved to a CSV file.
            cache_filename (str, optional): Name of the cache file to save the labeled data. Default is None.
                If None, the cache file will not be saved.            
            
        Returns:
            x_train (np.ndarray): The training data embeddings.
            y_train (np.ndarray): The training labels.
        """

        if dataset is not None:
            if isinstance(dataset, str) and os.path.isdir(dataset):

                # Get list of subfolders in the dataset folder
                path = filter(lambda el: os.path.isdir(os.path.join(dataset, el)), os.listdir(dataset))
                subfolders = sorted(path)

                # If there are no subfolders, we assume it is an unlabeled dataset
                if len(subfolders) == 0:
                    # List all files with allowed extensions
                    files = []
                    for ext in cfg.ALLOWED_FILETYPES:
                        files.extend(Path(dataset).rglob(f"*{ext}"))
                    files = [str(f) for f in files]

                    # Create dataframe for unlabeled dataset
                    self.df = pd.DataFrame({'file': files, 'categories': ["Unlabeled"] * len(files)})

                # If there are subfolders, we assume it is a labeled dataset with categories names
                else:
                    files = []
                    categories = []
                    for folder in subfolders:
                        folder_path = os.path.join(dataset, folder)
                        # List all files with allowed extensions
                        for ext in cfg.ALLOWED_FILETYPES:
                            for f in Path(folder_path).rglob(f"*{ext}"):
                                files.append(str(f))
                                categories.append(folder)

                    # Create dataframe for training dataset
                    self.df = pd.DataFrame({'file': files, 'categories': categories})

            elif isinstance(dataset, str) and dataset.lower().endswith('.csv') and os.path.isfile(dataset):
                self.df = pd.read_csv(dataset, sep=";")
            else:
                raise ValueError("Provided dataset path is not a directory or a valid CSV file.")
        else:
            # If self.df is already a DataFrame, do nothing; otherwise, get the dataset using the Xeno-Canto parent class
            if not isinstance(getattr(self, 'df', None), pd.DataFrame):
                self.df = self.data(csv_filename=csv_filename)

        # Get unique categories from the dataframe
        unique_categories = self.df['categories'].unique()
        categories = list(sorted(unique_categories))

        # Get valid labels
        valid_labels = [label for label in categories if label.lower() not in cfg.NON_EVENT_CLASSES and not label.startswith("-")]

        # Check if data is unlabeled or not
        if any(label != "Unlabeled" for label in valid_labels):
            # Check if binary classification
            cfg.BINARY_CLASSIFICATION = len(valid_labels) == 1

            # Validate the classes for binary classification
            if cfg.BINARY_CLASSIFICATION:
                if len([f for f in categories if f.startswith("-")]) > 0:
                    raise Exception(
                        "Negative labels can't be used with binary classification",
                        "validation-no-negative-samples-in-binary-classification",
                    )
                if len([f for f in categories if f.lower() in cfg.NON_EVENT_CLASSES]) == 0:
                    raise Exception(
                        "Non-event samples are required for binary classification",
                        "validation-non-event-samples-required-in-binary-classification",
                    )

            # Check if multi label
            cfg.MULTI_LABEL = len(valid_labels) > 1 and any("," in f for f in categories)

            # Check if multi-label and binary classficication
            if cfg.BINARY_CLASSIFICATION and cfg.MULTI_LABEL:
                raise Exception("Error: Binary classfication and multi-label not possible at the same time")

            # Only allow repeat upsampling for multi-label setting
            if cfg.MULTI_LABEL and cfg.UPSAMPLING_RATIO > 0 and cfg.UPSAMPLING_MODE != "repeat":
                raise Exception(
                    "Only repeat-upsampling ist available for multi-label", "validation-only-repeat-upsampling-for-multi-label"
                )

        # Get configuration 
        config = cfg.get_config()
        # Create a new dataframe for labeling
        df_labeling = pd.DataFrame()

        # Use custom classifier to label data
        if custom_classifier is not None:
            config['CUSTOM_CLASSIFIER'] = custom_classifier
            # Get the labels of the custom classifier
            filename, _ = os.path.splitext(custom_classifier)
            config['LABELS_FILE'] = filename + "_Labels.txt"
            labels = Path(config['LABELS_FILE']).read_text(encoding="utf-8").splitlines() if config['LABELS_FILE'] else []
            config['LABELS'] = labels

        for cat in categories:
            # Get list of files
            files = self.df[self.df['categories'] == cat]['file'].tolist()
            # Create dataframe to store the labels of single categories
            df_labels = pd.DataFrame()

            # Load files using thread pool
            with Pool(cfg.CPU_THREADS) as p:
                tasks = []

                # Load and split the audio files
                for f in files:
                    task = p.apply_async(partial(self.load_and_split, f=f, config=config))
                    tasks.append(task)

                # Wait for tasks to complete and monitor progress with tqdm
                with tqdm.tqdm(
                    total=len(tasks),
                    desc=f"Loading '{cat}' species" if self.verbose else None,
                    unit="f"
                ) as progress_bar:
                    # Iterate over the tasks and get the results
                    for task in tasks:
                        result = task.get()

                        # Check if the result is not empty
                        if len(result[0]) > 0:
                            df_batch = pd.DataFrame()
                            # Add categories to the dataframe
                            df_batch['categories'] = [cat] * len(result[0])
                            # Convert embeddings to numpy array and add to dataframe
                            emb_arr = np.asarray(result[0])
                            for i in range(emb_arr.shape[1]):
                                df_batch['emb' + str(i+1).zfill(4)] = emb_arr[:, i]
                            # Add timestamps and file paths to the dataframe
                            df_batch['start'] = [t[0] for t in result[1]]
                            df_batch['end'] = [t[1] for t in result[1]]
                            df_batch['file'] = result[2]

                            if custom_classifier is not None:
                                # Add predictions and confidences to the dataframe
                                df_batch['prediction'] = result[3]
                                df_batch['confidence'] = result[4]

                            # Concatenate the batch dataframe with the labels of single categories
                            df_labels = pd.concat([df_labels, df_batch], ignore_index=True)

                        # Update progress bar
                        progress_bar.update(1)

                    if custom_classifier is None:
                        # Label the data using density-based clustering if no custom classifier is provided
                        df_labels = cluster(df_labels, config, self.verbose)
                        # Keep only data pseudo-labeled as signal
                        df_labels = df_labels[df_labels.pseudo_label == 1]

                # Concatenate dataFrames
                df_labeling = pd.concat([df_labeling, df_labels], ignore_index=True)

        # Keep only categories of interest
        self.df = df_labeling[df_labeling.categories.isin(unique_categories)]
        # Rearrange index to be sure to have unique and increasing index
        self.df.reset_index(drop=True, inplace=True)

        # Get the embeddings
        x_train = self.df.loc[:, self.df.columns.str.startswith("emb")].to_numpy()
        if custom_classifier is not None:
            # Perform one-hot encoding the predictions of the custom classifier
            one_hot_encoded = pd.get_dummies(self.df.prediction.astype(str))
        else:
            # Perform one-hot encoding on categories
            one_hot_encoded = pd.get_dummies(self.df.categories.astype(str))
        # Get the labels
        y_train = one_hot_encoded.values

        if csv_filename is not None:            
            if self.save_path is not None:
                # Format save_path into Path
                save_path = Path(self.save_path)
                # Save and append dataframe 
                csv_filename = save_path / csv_filename

            if self.verbose:
                print(f"Saving training data to csv: {csv_filename}", flush=True)

            # Save dataframe
            self.df.to_csv(csv_filename, sep=";", encoding='utf-8', index=False, index_label='id') 

        if cache_filename is not None:
            if self.save_path is not None:
                # Format save_path into Path
                save_path = Path(self.save_path)
                # Save and append dataframe 
                cache_filename = save_path / cache_filename

            if self.verbose:
                print(f"Saving training data to cache: {cache_filename}", flush=True)

            # Save cache file with training data, test data, labels and configuration
            np.savez(
                cache_filename,
                x_train=np.array(x_train, dtype="float32"),
                y_train=np.array(y_train, dtype="float32"),
                labels=categories,
                binary_classification=cfg.BINARY_CLASSIFICATION,
                multi_label=cfg.MULTI_LABEL,
                fmin=cfg.BANDPASS_FMIN,
                fmax=cfg.BANDPASS_FMAX,
                crop_mode=cfg.SAMPLE_CROP_MODE,
                overlap=cfg.SIG_OVERLAP,
            )

        return np.array(x_train, dtype="float32"), np.array(y_train, dtype="float32")
    
    
    def load_and_split(self, f, config):
        """
        Load and split audio files to extract features and/or make predictions.

        Args:
            f: Path to the audio file.
            config: Configuration settings.

        Returns:
            embeddings: List of feature embeddings extracted from the audio file.
            timestamps: List of timestamps corresponding to the audio segments.
            file_paths: List of file paths for the audio segments.
            predictions: List of predicted labels (if predict is True).
            confidences: List of confidence scores for the predictions (if predict is True).
        """

        timestamps = []
        file_paths = []
        embeddings = []
        predictions = []
        confidences = []

        # Set configuration parameters
        cfg.set_config(config)

        # Try to open the audio file
        try:
            # Open audio file with FFmpeg
            sig, rate = self.open(
                f,
                duration=cfg.SIG_LENGTH if cfg.SAMPLE_CROP_MODE == "first" else None,
                fmin=cfg.BANDPASS_FMIN,
                fmax=cfg.BANDPASS_FMAX
            )

        # if anything happens print the error and ignore the file
        except Exception as e:
            # Print Error
            print(f"\t Error when loading file {f}", flush=True)
            print(f"\t {e}", flush=True)
            return np.array([]), np.array([])

        # Crop training samples
        if cfg.SAMPLE_CROP_MODE == "center":
            sig_splits = [audio.crop_center(sig, rate, cfg.SIG_LENGTH)]
        elif cfg.SAMPLE_CROP_MODE == "first":
            sig_splits = [audio.split_signal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)[0]]
        elif cfg.SAMPLE_CROP_MODE == "smart":
            # Detect peaks in audio energy to identify potential signals
            sig_splits = audio.smart_crop_signal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)
        else:
            sig_splits = audio.split_signal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)

        # Set audio timestamps
        start = 0
        end = start + cfg.SIG_LENGTH
        
        # Get feature embeddings and predictions
        batch_size = 1  # turns out that batch size 1 is the fastest, probably because of having to resize the model input when the number of samples in a batch changes

        for i in range(0, len(sig_splits), batch_size):

            # Get timestamps and file paths
            timestamps.append([start, end])
            file_paths.append(f)
            # Advance start and end
            start += cfg.SIG_LENGTH - cfg.SIG_OVERLAP
            end = start + cfg.SIG_LENGTH

            # Extract embeddings
            batch_sig = sig_splits[i : i + batch_size]
            e = model.embeddings(batch_sig)
            # Add to training data
            embeddings.extend(e)

            if cfg.CUSTOM_CLASSIFIER is not None:
                import operator
                # Predict labels
                p = model.predict(batch_sig)[0]
                # Assign scores to labels
                p_labels = zip(cfg.LABELS, p)
                # Sort by score
                p_sorted = sorted(p_labels, key=operator.itemgetter(1), reverse=True)
                # Add top prediction and confidence to lists
                predictions.append(p_sorted[0][0])  # Species name
                confidences.append(p_sorted[0][1])  # Confidence score

        return (embeddings, timestamps, file_paths, predictions, confidences) if cfg.CUSTOM_CLASSIFIER is not None else (embeddings, timestamps, file_paths)



if __name__ == "__main__":

    # Create an instance of the XCNet class
    xcnet = XCNet(save_path='train_data')
    
    # Build dataset from Xeno-Canto and save to csv
    xcnet.data(csv_filename='xc_dataset.csv')
    # Label data with clustering and save to cache
    xcnet.labeling(cache_filename="train_cache.npz")

    # # Download audio files from the training dataset
    # xcnet.download()

    # xcnet.train(custom_classifier='xcnet.tflite', cache_mode="load", cache_file="train_cache.npz")

    # xcnet = XCNet(save_path='test_data', overwrite=False)
    # x, y = xcnet.labeling(dataset='test_data', custom_classifier='train_data/xcnet.tflite')