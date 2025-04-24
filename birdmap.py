import os, shutil
import numpy as np
import pandas as pd
from pathlib import Path

from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import config as cfg
import model
from analyze import getRawAudioFromFile
from utils import writeErrorLog
from dataset import XenoCanto 


class BirdMAP(XenoCanto):
    
    def __init__(
                self,
                params_xc = cfg.PARAMS_XC,
                species_xc = cfg.SPECIES_XC,
                num_files = cfg.NUM_FILES,
                format_time = False,
                format_date = False,
                verbose = True,
                overwrite = False,
                custom_classifier_path = None,
                save_path = "train_data",
                ):

        self.params_xc = params_xc
        self.species_xc = species_xc
        self.num_files = num_files
        self.format_time = format_time
        self.format_date = format_date 
        self.verbose = verbose
        self.overwrite = overwrite
        self.custom_classifier_path = custom_classifier_path
        self.save_path = save_path
        
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
                

    def embeddings(self, dataset=None, csv_filename=None): 

        """
        Perform multi-CPU embeddings extraction of audio chunks.
        This function extract embeddings from a dataset of audio files using multiple CPU cores for parallel processing.

        Args:
            dataset (str, optional): Path to dataset. This could be the path to a pandas DataFrame or an audio recordings folder. Defaults to None.
            csv_filename (str, optional): The name of the csv file to save the embeddings. Default is None.

        Returns:
            pd.DataFrame: DataFrame containing all embeddings extracted for each ROI in the dataset.
        """
        
        if dataset is not None:
            self.df = self._load_dataframe(dataset)
        
        # Check for audio splitting duration
        duration = cfg.FILE_SPLITTING_DURATION
        
        if (self.df['length'] > duration).any():
            # Create a temporary dataframe
            df_temp = pd.DataFrame()
            
            for _, row in self.df.iterrows():
                if row.length > duration:
                    # Create new DataFrame with audio chunks
                    n_rows = int(row.length / duration) 
                    self.df = pd.concat([pd.DataFrame([row])] * n_rows, ignore_index=True)
                    # Add columns
                    self.df['length'] = range(duration, duration * len(self.df) + duration, duration)
                    self.df['offset'] = range(0, duration * len(self.df), duration)
                        
                # Concatenate dataframes    
                df_temp = pd.concat([self.df.reset_index(drop=True), df_temp.reset_index(drop=True)], ignore_index=True)
            
            self.df = df_temp
        
        # Check for potential unknown categories    
        if 'categories' not in self.df.columns:
            self.df['categories'] = "Unknown"

        # Extract embeddings using BirdNET 
        #-----------------------------
        # Create empty dataframe to contain all embeddings extracted for each audio chunk
        df_embeddings = pd.DataFrame()
        
        # Multi-CPU process
        #-------------------        
        if self.verbose:
            print('\n====================== EXTRACT EMBEDDINGS ======================\n')
        
        with tqdm(total=len(self.df), desc="Extract embeddings") as pbar:
            with ProcessPoolExecutor(max_workers=cfg.CPU_THREADS-1) as pool:
                try:
                    for df_embeddings_temp in pool.map(partial(self._extract_features), (row for _, row in self.df.iterrows())):
                        # Concatenate dataFrames
                        df_embeddings = pd.concat([df_embeddings, df_embeddings_temp], ignore_index=True)
                        # Update progress bar
                        pbar.update(1)   
                except ValueError:
                    pass 

        # Test if embeddings were extracted
        if len(df_embeddings) > 0:
            # Rearrange index to be sure to have unique and increasing index
            df_embeddings.reset_index(drop=True, inplace=True)
            self.df = df_embeddings

        if self.verbose:
            print(f'Number of audio chunks extracted: {len(self.df)}')

        if csv_filename is not None:
            self._save_dataframe(csv_filename)
        
        return self.df
    

    def pseudo_label(self, dataset=None, display=False, thumbnails=True, noise=False, csv_filename=None):

        """
        Data pseudo-labeling with density-based clustering method (DBCM).

        Data pseudo-labeling is performed with DBCM for several reasons:
            * DBCM does not need the number of clusters to do the clustering
            * DBCM is able to deal with noise and keep them outside any clusters.

        The goal of the clustering is to aggregate similar audio chunks which might correspond to the main call or song of a species. 
        If several clusters are found, which means that we might have ROIs corresponding to different calls and/or songs 
        for the species, we can keep the cluster with the highest number of ROIs or all the clusters.
        
        Args:
            dataset (str, optional): Path to dataset. This could be the path to a pandas DataFrame or an audio recordings folder. Defaults to None.
            display (bool, optional): Whether to display the clusters or not. The default is False.
            thumbnails (bool, optional): Whether to display thumbnails of the clusters or not. The default is True.
            noise (bool, optional): Whether to display noise or not. The default is False.
            csv_filename (str, optional): The name of the csv file to save the pseudo-labeled data. Default is None.

        Returns:
            pd.DataFrame: DataFrame containing all the pseudo-labeled data.
        """

        if self.verbose:
            print(f'\n====================== PSEUDO-LABELING DATA WITH {cfg.METHOD} ======================\n')    
        
        if dataset is not None:
            self.df = self._load_dataframe(dataset)
        
        # Count the categories
        count = 0
            
        # Find the cluster for each categories separately
        #-------------------------------------------------------  
        for categories in np.sort(self.df.categories.unique()):

            # Select the ROIs of the current categories
            df_single_categories = self.df[self.df["categories"] == categories]
            
            # Test if the number of ROIs is higher than 2.
            # If not, it is impossible to cluster ROIs. It requires at least 3 ROIS
            if len(df_single_categories) < 3:
                self.df["cluster_number"] = -1 # noise
                self.df["pseudo_label"] = 0 # noise
                
                if self.verbose:
                    print("Only {} ROIs. It requires at least 3 ROIs to perform clustering".format(len(df_single_categories)))
                    
            else:
                # Prepare the embeddings of that categories
                #-------------------------------------------------------
                config = cfg.get_config()

                from cluster import estimate_parameters
                X = df_single_categories.loc[:, df_single_categories.columns.str.startswith("emb")].to_numpy() 

                print(f'Clustering {count+1}/{len(self.df.categories.unique())} - {categories} species\n')

                if cfg.MIN_PTS == 'auto':
                    # Automatic estimation of the clustering algorithm parameters
                    #--------------------------------------------------------------------
                    min_samples = 3
 
                    if self.verbose:
                        print(f'Automatic estimation of the clustering algorithm parameters...')
                    
                    dbscan_parameters = []
                    best_score = None
                    
                    # Multi-CPU process
                    #-------------------                    
                    with ProcessPoolExecutor(max_workers=cfg.CPU_THREADS-1) as executor:
                        while True:
                            future = executor.submit(estimate_parameters, X, config, min_samples, cfg.METRIC)
                            results = future.result()
                            if not np.isnan(results[0]):  
                                dbscan_parameters.append(results)
                                
                                # Check if the current score is lower than the previous best score
                                if best_score is not None and results[0] < best_score[0]:
                                    break
                                # Update the best score
                                best_score = results
                            # Increment the number of components
                            min_samples += 1
                        
                    # Use the best score found
                    dbcv = best_score[0]
                    cluster = best_score[1] 
                    eps = best_score[2]
                    min_samples = best_score[3]
 
                else:       
                    if cfg.MIN_PTS > 1:
                        min_samples = round(cfg.MIN_PTS)
                    else:
                        min_samples = 3 

                    dbcv, cluster, eps, min_samples = estimate_parameters(X, config, min_samples, cfg.METRIC)

                if self.verbose:                
                    # Get the predicted labels and the number of clusters (excluding noise)
                    #---------------------------------------------------
                    clusters = len(np.unique(cluster.labels_[cluster.labels_ >= 0])) 
                    noisy_percentage = round(np.sum(cluster.labels_ == -1) / X.shape[0], 2) * 100           
                    print(f'Number of clusters found (excluding noise): {clusters} (min_samples={min_samples}, eps={eps})')
                    print(f'Percentage of noise/outliers data: {noisy_percentage}%')
                    print(f'Density-Based Clustering Validation: {round(dbcv,4)}\n') 

                # Add the label found with the clustering
                #-------------------------------------------------------
                self.df.loc[self.df["categories"] == categories, "cluster_number"] = cluster.labels_.reshape(-1, 1)
        
                # Add the automatic label (SIGNAL = 1 or NOISE = 0) into the pseudo_label's column of the dataframe
                # Test if we want to consider only the biggest or all clusters that are not noise (-1) to be signal
                
                if cfg.KEEP == 'BIGGEST':
                    # Set by default to 0 the pseudo_label of all
                    self.df.loc[self.df["categories"] == categories, "pseudo_label"] = int(0)
                    # Find the cluster ID of the biggest cluster that is not noise
                    try:
                        biggest_cluster_ID = self.df.loc[(self.df["categories"] == categories) & (
                                                        self.df["cluster_number"] >= 0)]["cluster_number"].value_counts().idxmax()

                        # Set to 1 the pseudo_label of the biggest cluster
                        self.df.loc[(self.df["categories"] == categories) & (
                                        self.df["cluster_number"] == biggest_cluster_ID), "pseudo_label"] = int(1)
                    except:
                        # If there is only noise
                        pass
                    
                elif cfg.KEEP == 'ALL':                
                    # Set by to 0 the pseudo_label of the noise (cluster ID = -1)
                    self.df.loc[(self.df["categories"] == categories) & (
                                    self.df["cluster_number"] < 0), "pseudo_label"] = int(0)
                    # Set by to 1 the pseudo_label of the signal (cluster ID >= 0)
                    self.df.loc[(self.df["categories"] == categories) & (
                                    self.df["cluster_number"] >= 0), "pseudo_label"] = int(1) 
                    
                    # If there is only one category we can analyze the repertoire of the species
                    if len(self.df.categories.unique()) == 1:
                        # Add the cluster number to the categories to differentiate them
                        self.df.categories = self.df.apply(
                            lambda row: f"{row['categories']}{row['cluster_number']}", axis=1)   

                if display:
                    if self.save_path is not None:
                        # Format save_path into Path
                        save_path = Path(self.save_path)
                        
                    from display import interactive_display
                    interactive_display(self.df, X=X, output_filename=categories, save_path=save_path, thumbnails=thumbnails, noise=noise)
                
            # Increment
            count += 1
                
        # Default name of the csv file with the cluster
        #-------------------------------------------------------------------------
        if csv_filename is not None:
            self._save_dataframe(csv_filename)

        return self.df

    
    def train(self, dataset=None, on_epoch_end=None, on_trial_result=None, autotune_directory="autotune", model_output_path="checkpoints/custom/XCNet.tflite", csv_filename=None):
        
        """
        Train custom classifier on top of BirdNET model with pseudo-labeled data requested from the Xeno-Canto database.

        Args:
            dataset (str, optional): Path to dataset. This could be the path to a pandas DataFrame or audio recordings folder. Defaults to None (which corresponds to self.df).
            on_epoch_end: A callback function that takes two arguments `epoch`, `logs`.
            model_output_path (str, optional): Path to the output custom classifier. Default is "checkpoints/custom/XCNet.tflite".
            csv_filename (str, optional): The name of the csv file to save the dataset. Default is None.

        Returns:
            A keras `History` object, whose `history` property contains all the metrics.
        """

        if dataset is not None:
            self.df = self._load_dataframe(dataset)

        # Check if dataset contains already pseudo labels or not
        if 'pseudo_label' not in self.df.columns:                
            # Build a pseudo-labeled dataset with density-based clustering method       
            self.df = self.embeddings() # extract embeddings from BirdNET model first              
            # Find cluster and assign pseudo-labels
            self.df = self.pseudo_label(csv_filename=csv_filename) 

        # Keep only data pseudo-labeled as signal
        self.df = self.df[self.df.pseudo_label == 1].reset_index(drop=True)

        if self.verbose:
            print('\n====================== TRAINING THE MODEL ======================\n')
            print("Loading training data...", flush=True)
        
        # Get the embeddings
        x_train = self.df.loc[:, self.df.columns.str.startswith("emb")].to_numpy()

        # Get the unique categories
        labels = self.df.categories.unique()        
        # Perform one-hot encoding on categories
        one_hot_encoded = pd.get_dummies(self.df.categories) 
        y_train = one_hot_encoded.values
        
        # Convert to NumPy array         
        x_train = np.array(x_train, dtype="float32")
        y_train = np.array(y_train, dtype="float32")
            
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
            model.save_raven_model(classifier, model_output_path, labels)
            model.saveLinearClassifier(classifier, model_output_path, labels, mode=cfg.TRAINED_MODEL_SAVE_MODE)
        elif cfg.TRAINED_MODEL_OUTPUT_FORMAT == "tflite":
            model.saveLinearClassifier(classifier, model_output_path, labels, mode=cfg.TRAINED_MODEL_SAVE_MODE)
        elif cfg.TRAINED_MODEL_OUTPUT_FORMAT == "raven":
            model.save_raven_model(classifier, model_output_path, labels)
        else:
            raise ValueError(f"Unknown model output format: {cfg.TRAINED_MODEL_OUTPUT_FORMAT}")

        print(f"Done! Best AUPRC: {best_val_auprc}, Best AUROC: {best_val_auroc}", flush=True)
        print(f"Path to trained classifier model output: {model_output_path}", flush=True)

        # Update model and label file paths
        cfg.MODEL_PATH = model_output_path
        filename, _ = os.path.splitext(model_output_path)
        cfg.LABELS_FILE = f"{filename}_Labels.txt"

        return history


    def _extract_features(self, row):
        
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
                chunks = getRawAudioFromFile(fpath, offset, cfg.FILE_SPLITTING_DURATION)
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
                    f = model.features(data)
                    # Append features to list
                    embeddings.append(f[0])

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
            writeErrorLog(ex)
            return
        
        try:
            # Save embeddings
            return df

        except Exception as ex:
            # Write error log
            print(f"Error: Cannot save embeddings for {fpath}.", flush=True)
            writeErrorLog(ex)
            return


    def _load_dataframe(self, dataset):
        # Check if dataset is a path to a dataframe, audio file or folder
        if type(dataset) == str:
            ext = os.path.splitext(dataset)[1]
            # Check file extension
            if ext == ".csv": 
                self.df = pd.read_csv(dataset, sep=";")
            else:
                import audio
                # If it's not a csv file, we assume it's an audio file or a folder with audio files
                self.df = audio.saveAudioAsDataframe(dataset)
        else:
            self.df = dataset

        return self.df
    

    def _save_dataframe(self, csv_filename):
        if self.save_path is not None:
            # Format save_path into Path
            save_path = Path(self.save_path)
            # Get full path to the csv file 
            csv_filename = save_path / csv_filename
        # Save and append dataframe
        self.df.to_csv(csv_filename, sep=";", encoding='utf-8', index=False, index_label='id')



           

if __name__ == "__main__":
    
    ### TRAIN
    mapper = BirdMAP(save_path='train_data') # create Xeno-Canto network with 'train_data' folder to save data

    # We use specific clustering settings for building pseudo-labeled dataset
    config = cfg.get_config()
    config["KEEP"] = 'BIGGEST'
    cfg.set_config(config) # restore config in case we're on Windows to be thread save

    # Build Xeno-Canto dataset with Xeno-Canto parameters from the config file (config.py)
    mapper.dataset(csv_filename="xc_dataset.csv")     
    # Extract embeddings from BirdNET model
    mapper.embeddings(csv_filename='embeddings.csv')   
    # Find cluster and assign pseudo-labels
    mapper.pseudo_label(csv_filename='pseudo_label.csv')

    # Train model with Xeno-Canto pseudo-labeled dataset 
    mapper.train(model_output_path="checkpoints/custom/XCNet.tflite", csv_filename='train_dataset.csv')


    # ### TEST
    # mapper = BirdMAP(save_path='test_data', custom_classifier_path='checkpoints/custom/XCNet.tflite')

    # # We use specific clustering settings for extracting the clusters
    # config = cfg.get_config()
    # config["KEEP"] = 'ALL'
    # cfg.set_config(config) # restore config in case we're on Windows to be thread save

    # # Extract embeddings from audio files in 'test_data' folder with custom XCNet
    # mapper.embeddings(dataset='test_data/', csv_filename='embeddings.csv') 
    # # Pseudo-label data and display clusters
    # mapper.pseudo_label(dataset='test_data/embeddings.csv', csv_filename='pseudo_label.csv', display=True)  





    # if len(xcnet.df.categories.unique()) == 1: 
    #     config["KEEP"] = 'ALL' # keep all clusters for repertoire analysis on an unique species