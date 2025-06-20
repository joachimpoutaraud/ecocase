import os
import shutil
import pandas as pd
import numpy as np

import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from birdnet_analyzer import config as cfg
from birdnet_analyzer import model

from dataset import XenoCanto
from audio import split_audio


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

    def labeling(self, dataset=None, method='clustering', keep=cfg.KEEP, noise=False, csv_filename=None, save_cache=None):
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
            noise (bool, optional): If True, the data labeled as noise will be kept. Default is False.
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

        # # Get valid labels
        # valid_labels = [label for label in categories if label.lower() not in cfg.NON_EVENT_CLASSES and not label.startswith("-")]

        # # Check if data is unlabeled or not
        # if any(label != "Unlabeled" for label in valid_labels):
        #     # Check if binary classification
        #     cfg.BINARY_CLASSIFICATION = len(valid_labels) == 1

        #     # Validate the classes for binary classification
        #     if cfg.BINARY_CLASSIFICATION:
        #         if len([f for f in categories if f.startswith("-")]) > 0:
        #             raise Exception(
        #                 "Negative labels can't be used with binary classification",
        #                 "validation-no-negative-samples-in-binary-classification",
        #             )
        #         if len([f for f in categories if f.lower() in cfg.NON_EVENT_CLASSES]) == 0:
        #             raise Exception(
        #                 "Non-event samples are required for binary classification",
        #                 "validation-non-event-samples-required-in-binary-classification",
        #             )

        #     # Check if multi label
        #     cfg.MULTI_LABEL = len(valid_labels) > 1 and any("," in f for f in categories)

        #     # Check if multi-label and binary classficication
        #     if cfg.BINARY_CLASSIFICATION and cfg.MULTI_LABEL:
        #         raise Exception("Error: Binary classfication and multi-label not possible at the same time")

        #     # Only allow repeat upsampling for multi-label setting
        #     if cfg.MULTI_LABEL and cfg.UPSAMPLING_RATIO > 0 and cfg.UPSAMPLING_MODE != "repeat":
        #         raise Exception(
        #             "Only repeat-upsampling ist available for multi-label", "validation-only-repeat-upsampling-for-multi-label"
        #         )

        # Create a new dataframe for labeling
        df_labeling = pd.DataFrame()
        # Get configuration parameters
        config = cfg.get_config()

        # Iterate over the categories
        count = 1
        # Iterate over the categories
        for cat in categories:
            # Get list of files
            files = self.df[self.df['categories'] == cat]['file'].tolist()
            # Create dataframe to store the labels of single categories
            df_labels = pd.DataFrame()

            # Parallelize audio segmentation with split_audio
            with ProcessPoolExecutor(max_workers=cfg.CPU_THREADS) as executor:
                split_futures = {
                    executor.submit(split_audio, f): f for f in files
                }

                with tqdm.tqdm(
                    total=len(split_futures),
                    desc=f"Segmenting ROIs for '{cat}' species ({str(count)}/{str(len(categories))})" if self.verbose else None,
                    unit="f"
                ) as progress_bar:
                    for future in as_completed(split_futures):
                        f = split_futures[future]
                        try:
                            df_split = future.result()
                            if isinstance(df_split, pd.DataFrame) and not df_split.empty:
                                # Add columns from self.df where 'file' matches
                                row_data = self.df[self.df['file'] == f]
                                missing_cols = [col for col in self.df.columns if col not in df_split.columns]
                                # Create a DataFrame with missing columns
                                if not row_data.empty and missing_cols:
                                    prefix_df = pd.DataFrame({col: row_data.iloc[0][col] for col in missing_cols}, index=df_split.index)
                                    df_split = pd.concat([prefix_df, df_split], axis=1)
                                # Concatenate to df_labels
                                df_labels = pd.concat([df_labels, df_split], ignore_index=True)
                        except Exception as e:
                            if self.verbose:
                                print(f"Error segmenting {f}: {e}")
                        progress_bar.update(1)

            # Get list of raw audio files from the segmented dataframe
            files = df_labels['raw_audio'].tolist()
            # Pre-allocate embedding columns
            emb_dim = None
            # Try to get embedding dimension from a sample
            if files:
                sample_emb = self.extract_and_predict(files[0], config, method != 'clustering')
                emb_dim = np.asarray(sample_emb[0]).shape[-1]
                for i in range(emb_dim):
                    col = f'emb{i+1:04d}'
                    if col not in df_labels.columns:
                        df_labels[col] = np.nan
                if method != 'clustering':
                    if 'prediction' not in df_labels.columns:
                        df_labels['prediction'] = ""
                    if 'confidence' not in df_labels.columns:
                        df_labels['confidence'] = np.nan

            # Parallelize embedding extraction and prediction over df_labels rows
            with ProcessPoolExecutor(max_workers=cfg.CPU_THREADS) as executor:
                extract_futures = {
                    executor.submit(self.extract_and_predict, f, config, method != 'clustering'): idx
                    for idx, f in enumerate(files)
                }

                with tqdm.tqdm(
                    total=len(extract_futures),
                    desc=f"Extracting embeddings from BirdNET" if self.verbose else None,
                    unit="f"
                ) as progress_bar:
                    for future in as_completed(extract_futures):
                        idx = extract_futures[future]
                        try:
                            result = future.result()
                            emb_arr = np.asarray(result[0])
                            if emb_arr.ndim == 1:
                                emb_arr = np.expand_dims(emb_arr, axis=0)
                            # Assign embeddings to the row
                            for i in range(emb_arr.shape[1]):
                                df_labels.at[idx, f'emb{i+1:04d}'] = emb_arr[0, i]
                            if method != 'clustering':
                                df_labels.at[idx, 'prediction'] = result[1]
                                df_labels.at[idx, 'confidence'] = result[2]
                        except Exception as e:
                            if self.verbose:
                                print(f"Error extracting embeddings for {files[idx]}: {e}")
                        progress_bar.update(1)

            # Label data using density-based clustering if no custom classifier is provided
            if method == 'clustering':
                from clustering import cluster
                config['KEEP'] = keep.upper()
                df_labels = cluster(df_labels, config, self.verbose)

                if noise is False:
                    # Keep only data labeled as signal (i.e. not noise)
                    df_labels = df_labels[df_labels['pseudo_label'] == 1]

            # Concatenate dataFrames
            df_labeling = pd.concat([df_labeling, df_labels], ignore_index=True)
            # Increment categories count
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


    def extract_and_predict(self, y, config, predict=False):
        """
        This function extracts feature embeddings.
        It also predicts labels if a custom classifier is provided.

        Args:
            y: audio data.
            config: Configuration settings.
            predict: Boolean indicating whether to predict labels using a custom classifier. Default is False.

        Returns:
            embeddings: List of feature embeddings extracted from the audio file.
            timestamps: List of timestamps corresponding to the audio segments.
            file_paths: List of file paths for the audio segments.
            predictions: List of predicted labels (if predict is True).
            confidences: List of confidence scores for the predictions (if predict is True).
        """

        # Set configuration parameters
        cfg.set_config(config)
        
        # Extract embeddings
        y = np.expand_dims(y, axis=0)          
        embeddings = model.embeddings(y)

        if predict:
            import operator
            # Predict labels
            p = model.predict(y)[0]
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
            predictions = p_sorted[0][0] # Species name
            confidences = p_sorted[0][1] # Confidence score

        return (embeddings, predictions, confidences) if predict else embeddings
    

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

    # # Create an instance of the XCNet class
    # xcnet = XCNet(save_path='train_data')
    # # # Build dataset from Xeno-Canto and save to csv
    # # xcnet.dataset(csv_filename='xc_dataset.csv', download=True)
    # # Label data with clustering and save to cache
    # xcnet.labeling(dataset='train_data/xc_dataset.csv', method='clustering', keep='all', csv_filename="labels.csv")
    # # Plot the results
    # xcnet.plot()


    # # Train model with pseudo-labeled dataset and save
    # xcnet.train(load_cache='train_data/train_cache.npz', save_model='xcnet.tflite')


    # Label unlabeled dataset with custom classifier and plot the results
    xcnet = XCNet(save_path='test_data')
    # xcnet.labeling(dataset='test_data', method='classification', csv_filename='labels.csv')
    xcnet.labeling(dataset='test_data', method='clustering', keep='all', csv_filename='labels.csv')
    xcnet.plot()







    # from birdnet_analyzer.species.core import species
    # species("species_list.txt", lat=4.645498315764353, lon=-52.65625921753347, week=11)

    ###########################################################################################
    # TODO LIST
    # 1) Use default BirdNET labels for pseudo-labeling and train classifier to assign clusters
    # 2) Whether to include the data labeled as noise as a category when training the model?
    # 3) Improve parallel processing when loading ROIs in memory (not working at the moment)
    # 4) Add more color possibilities for the display function
    ###########################################################################################