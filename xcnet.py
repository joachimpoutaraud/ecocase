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


class XCNet(XenoCanto):

    def __init__(
                self,
                custom_classifier = cfg.CUSTOM_CLASSIFIER,
                params_xc = cfg.PARAMS_XC,
                species_xc = cfg.SPECIES_XC,
                num_files = cfg.NUM_FILES,
                format_time = False,
                format_date = False,
                verbose = True,
                save_path = None,
                overwrite = False,
                ):

        self.custom_classifier = custom_classifier
        self.params_xc = params_xc
        self.species_xc = species_xc
        self.num_files = num_files
        self.format_time = format_time
        self.format_date = format_date 
        self.verbose = verbose
        self.save_path = save_path
        self.overwrite = overwrite

        if self.custom_classifier is not None:
            config = cfg.get_config()
            config['CUSTOM_CLASSIFIER'] = self.custom_classifier
            cfg.set_config(config)

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


    def train(self, load_cache=cfg.TRAIN_CACHE_FILE, on_epoch_end=None, on_trial_result=None, on_data_load_end=None, autotune_directory="autotune", save_model=cfg.CUSTOM_CLASSIFIER):

        from birdnet_analyzer.train import utils
        
        config = cfg.get_config()

        if self.save_path is not None:
            # Format save_path into Path
            save_path = Path(self.save_path)
            # Save and append directory to model and cache file
            save_model = str(save_path / os.path.basename(save_model))
            load_cache = str(save_path / os.path.basename(load_cache))

            config['CUSTOM_CLASSIFIER'] = save_model
            config['TRAIN_CACHE_FILE'] = load_cache
        else:
            # If save_path is None, use the default paths
            config['CUSTOM_CLASSIFIER'] = save_model
            config['TRAIN_CACHE_FILE'] = load_cache

        if load_cache is not None:
            config['TRAIN_CACHE_MODE'] = "load"
        
        # Set the configuration parameters
        cfg.set_config(config)

        # Train the model
        utils.train_model(on_epoch_end=on_epoch_end, on_trial_result=on_trial_result, on_data_load_end=on_data_load_end, autotune_directory=autotune_directory)

        if self.verbose:
            print(f"\nSaving custom classifier: {save_model}", flush=True)


    def labeling(self, dataset=None, method='clustering', keep=cfg.KEEP, csv_filename=None, save_cache=None):
        """ 
        Data-labeling using clustering or classification. By default, it uses clustering to label the data.

        Args:   
            dataset (str or pd.DataFrame, optional): Path to the dataset. It can be a CSV file or a folder path containing audio files. Default is None.
                If None, the dataset is queried from Xeno-Canto according to the Xeno-Canto parameters in the configuration file.
            method (str, optional): Method to use for labeling. It can be 'clustering' or 'classification'. Default is 'clustering'.
                If 'clustering', the data will be labeled using density-based clustering (DBSCAN or HDBSCAN).
                If 'classification', the data will be labeled using predictions of a classifier (e.g. BirdNet or custom classifier).
            keep (str, optional): If 'biggest', only the biggest cluster will be kept as signal. If 'all', all clusters will be kept as signal. Default is 'BIGGEST'.
                This parameter is only used if method is set to 'clustering'.
            csv_filename (str, optional): Name of the CSV file to save the labeled data. Default is None.
                If None, the labeled data will not be saved to a CSV file.
            save_cache (str, optional): Name of the cache file to save the labeled data. Default is None.
                If None, the cache file will not be saved.            
            
        Returns:
            pd.DataFrame: DataFrame containing the labeled data with embeddings and labels.
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

        # Create a new dataframe for labeling
        df_labeling = pd.DataFrame()
        # Get configuration parameters
        config = cfg.get_config()

        count = 1
        # Iterate over the categories
        for cat in categories:
            # Get list of files
            files = self.df[self.df['categories'] == cat]['file'].tolist()
            # Create dataframe to store the labels of single categories
            df_labels = pd.DataFrame()

            # Load files using thread pool
            with Pool(cfg.CPU_THREADS) as p:
                tasks = []

                # Load and split the audio files to extract features and/or make predictions
                for f in files:
                    if method == 'clustering':
                        task = p.apply_async(partial(self.split_and_extract, f=f, config=config))
                    else:
                        task = p.apply_async(partial(self.split_and_extract, f=f, config=config, predict=True))
                    tasks.append(task)

                # Wait for tasks to complete and monitor progress with tqdm
                with tqdm.tqdm(
                    total=len(tasks),
                    desc=f"Loading '{cat}' species ({str(count)}/{str(len(categories))})" if self.verbose else None,
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

                            if method != 'clustering':
                                # Add predictions and confidences to the dataframe
                                df_batch['prediction'] = result[3]
                                df_batch['confidence'] = result[4]

                            # Concatenate the batch dataframe with the labels of single categories
                            df_labels = pd.concat([df_labels, df_batch], ignore_index=True)

                        # Update progress bar
                        progress_bar.update(1)

            # Label data using density-based clustering if no custom classifier is provided
            if method == 'clustering':
                from clustering import cluster
                config['KEEP'] = keep.upper()
                df_labels = cluster(df_labels, config, self.verbose)
                # Keep only data labeled as signal (i.e. not noise)
                df_labels = df_labels[df_labels['pseudo_label'] == 1]

            # Concatenate dataFrames
            df_labeling = pd.concat([df_labeling, df_labels], ignore_index=True)

            count += 1

        # Keep only categories of interest
        self.df = df_labeling[df_labeling.categories.isin(unique_categories)]
        # If there is only one category we analyze the repertoire of the species
        if keep.upper() == 'ALL' and len(categories) == 1:
            # Add the cluster number to the categories to differentiate them
            self.df.categories = self.df.apply(lambda row: f"{row['categories']}_{row['prediction']}", axis=1) 
        # Rearrange index to be sure to have unique and increasing index
        self.df.reset_index(drop=True, inplace=True)

        # Add embeddings and labels to the class instance
        self.embeddings = self.df.loc[:, self.df.columns.str.startswith("emb")].to_numpy()
        
        if method == 'clustering':
            # If clustering, use categories as labels
            self.labels = self.df.categories.astype(str)
        else:
            # If classification, use predictions as labels
            self.labels = self.df.prediction.astype(str)

        if csv_filename is not None:            
            if self.save_path is not None:
                # Format save_path into Path
                save_path = Path(self.save_path)
                # Save and append dataframe 
                csv_filename = save_path / csv_filename

            if self.verbose:
                print(f"Saving labeled data to csv: {csv_filename}", flush=True)

            # Save dataframe
            self.df.to_csv(csv_filename, sep=";", encoding='utf-8', index=False, index_label='id') 

        if save_cache is not None:
            if self.save_path is not None:
                # Format save_path into Path
                save_path = Path(self.save_path)
                # Save and append dataframe 
                save_cache = save_path / save_cache

            # Set cache filename
            config = cfg.get_config()
            config['TRAIN_CACHE_FILE'] = save_cache
            config = cfg.set_config(config)

            if self.verbose:
                print(f"Saving training data to cache: {cfg.TRAIN_CACHE_FILE}", flush=True)

            # Encode labels according to birdnet_analyzer format
            one_hot_encoded = pd.get_dummies(self.labels)

            np.savez(
                save_cache,
                x_train=np.array(self.embeddings, dtype="float32"),
                y_train=np.array(one_hot_encoded.values, dtype="float32"),
                labels=categories,
                binary_classification=cfg.BINARY_CLASSIFICATION,
                multi_label=cfg.MULTI_LABEL,
                fmin=cfg.BANDPASS_FMIN,
                fmax=cfg.BANDPASS_FMAX,
                crop_mode=cfg.SAMPLE_CROP_MODE,
                overlap=cfg.SIG_OVERLAP,
            )

        return self.df
    

    def split_and_extract(self, f, config, predict=False):
        """
        This function opens an audio file, splits it into segments, and extracts feature embeddings.
        It also predicts labels if a custom classifier is provided.

        Args:
            f: Path to the audio file.
            config: Configuration settings.
            predict: Boolean indicating whether to predict labels using a custom classifier. Default is False.

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

            if predict:
                import operator

                # Predict labels
                p = model.predict(batch_sig)[0]
                # Logits or sigmoid activations?
                if cfg.APPLY_SIGMOID:
                    p = model.flat_sigmoid(np.array(p), sensitivity=-cfg.SIGMOID_SENSITIVITY)

                # Get the labels file
                if self.custom_classifier is not None:
                    # Get the labels of the custom classifier
                    filename = os.path.splitext(self.custom_classifier)[0]
                    # Here we assume that the labels file is in the same directory as the custom classifier
                    cfg.LABELS_FILE = filename + "_Labels.txt" 

                # Assign scores to labels
                labels = Path(cfg.LABELS_FILE).read_text(encoding="utf-8").splitlines() if cfg.LABELS_FILE else []
                p_labels = zip(labels, p)
                # Sort by score
                p_sorted = sorted(p_labels, key=operator.itemgetter(1), reverse=True)
                # Add top prediction and confidence to lists
                predictions.append(p_sorted[0][0])  # Species name
                confidences.append(p_sorted[0][1])  # Confidence score

        return (embeddings, timestamps, file_paths, predictions, confidences) if predict else (embeddings, timestamps, file_paths)
    
    
    def plot(self, dataset=None, min_confidence=cfg.MIN_CONFIDENCE, thumbnails=True, encode=False, frame_width=1000, frame_height=600, point_size=5, html_filename='scatterplot.html'):
        """
        Display the audio file and its spectrogram.

        Args:


        Returns:
            tuple: Paths to the saved spectrogram and audio files.
        """
        
        from display import interactive_display

        # Check if dataset is a path to a dataframe
        if dataset is not None:
            ext = os.path.splitext(dataset)[1]
            # Check file extension
            if ext == ".csv": 
                self.df = pd.read_csv(dataset, sep=";")
            else:
                raise ValueError("Provided dataset path is not a valid CSV file.")

        config = cfg.get_config()
        config['MIN_CONFIDENCE'] = min_confidence
        cfg.set_config(config)

        interactive_display(
            self.df, 
            thumbnails=thumbnails, 
            encode=encode, 
            frame_width=frame_width, 
            frame_height=frame_height, 
            point_size=point_size, 
            html_filename=html_filename, 
            save_path=self.save_path
            )
        

if __name__ == "__main__":

    # Create an instance of the XCNet class
    xcnet = XCNet(save_path='train_data')
    
    # Build dataset from Xeno-Canto and save to csv
    xcnet.data(csv_filename='xc_dataset.csv')
    # xcnet.download()
    # Label data with clustering and save to cache
    # xcnet.labeling(dataset='xc_dataset.csv', method='clustering', save_cache="train_cache.npz")
    # # Train model with pseudo-labeled dataset and save
    # xcnet.train(load_cache='train_data/train_cache.npz', save_model='xcnet.tflite')

    # # Label unlabeled dataset with custom classifier and plot the results
    # xcnet = XCNet(save_path='test_data')
    # xcnet.labeling(dataset='test_data', method='classification', csv_filename='labels.csv')
    # # xcnet.labeling(dataset='test_data', method='clustering', keep='all', csv_filename='labels.csv')
    # xcnet.plot()





    ###########################################################################################
    # TODO LIST
    # 1) Whether to include the data labeled as noise as a category when training the model?
    # 2) Update open function to bandpass filter and split audio files using FFmpeg
    ###########################################################################################