import os, shutil
import numpy as np
import pandas as pd
from pathlib import Path

from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import config as cfg
from xcset import XenoCanto
from birdnet import BirdNET  


class XCNet(XenoCanto, BirdNET):
    
    def __init__(
                self,
                params_xc = cfg.PARAMS_XC,
                species_xc = cfg.SPECIES_XC,
                format_time = False,
                format_date = False,
                verbose = True,
                overwrite = False,
                custom_classifier_path = None,
                save_path = cfg.TRAIN_DATA_PATH,
                ):

        self.params_xc = params_xc
        self.species_xc = species_xc
        self.format_time = format_time
        self.format_date = format_date 
        self.verbose = verbose
        self.overwrite = overwrite
        self.save_path = save_path
        
        # Check whether root directory is existing or not   
        if os.path.exists(self.save_path):
            if self.overwrite:
                # Remove directory
                shutil.rmtree(str(self.save_path))
                # Save dataframe
                os.makedirs(self.save_path)      
        else:      
            os.makedirs(self.save_path) 
            
        if custom_classifier_path is not None:
            filename, _ = os.path.splitext(custom_classifier_path)
            # Update model and label file paths
            cfg.MODEL_PATH = custom_classifier_path
            cfg.LABELS_FILE = f"{filename}_Labels.txt"
        
        
    def training(self, dataset=None, train_data_path=None, model_output_path=cfg.CUSTOM_CLASSIFIER, csv_filename=None):
        """
        Training custom classifier on top of BirdNET model with pseudo-labeled data requested from the Xeno-Canto database.
        
        Args:
            dataset (str, optional): Path to dataset. This could be the path to a pandas DataFrame or audio recordings folder. Defaults to None (which corresponds to self.df). 
            train_data_path (str, optional): Path to training dataset of labeled audio files, organized in folders by species (we use folder names as labels). This also works for non-bird species, as long as you have a dataset of labeled audio files. 
            model_output_path (str, optional): Path to the fine-tuned model. Default is cfg.CUSTOM_CLASSIFIER.
            csv_filename (str, optional): The name of the csv file to save the dataset. Default is None.
            
        Returns:
            A keras `History` object, whose `history` property contains all the metrics.
        """ 
        
        if dataset is not None:
        # Check if dataset is a path to a dataframe, audio file or folder
            if type(dataset) == str:
                ext = os.path.splitext(dataset)[1]
                # Check file extension
                if ext == ".csv": 
                    import pandas as pd
                    self.df = pd.read_csv(dataset, sep=";")
                else:
                    import audio
                    # If it's not a csv file, we assume it's an audio file or a folder with audio files
                    self.df = audio.saveAudioAsDataframe(dataset)
            else:
                self.df = dataset
                
        if train_data_path is not None:
            # Update training data path and train model
            cfg.TRAIN_DATA_PATH = train_data_path
        
        else: # we build a pseudo-labeled dataset with density-based clustering method       
            # Extract embeddings from BirdNET
            self.df = self.embeddings(csv_filename=csv_filename)
            
            # Find cluster and assign pseudo-labels
            c = cfg.getConfig() # get initial configuration dict
            # We use specific clustering settings for building pseudo-labeled dataset
            cfg.MIN_PTS, cfg.EPS, cfg.METHOD, cfg.KEEP = 3, 'auto', 'DBSCAN', 'BIGGEST' 
            self.df = self.pseudo_labeling(csv_filename=csv_filename) 
            # Reset config file with initial values
            cfg.MIN_PTS, cfg.EPS, cfg.METHOD, cfg.KEEP = c['MIN_PTS'], c['EPS'], c['METHOD'], c['KEEP'] 
            
            # Download dataset to disk
            from utils import download_audio_files
            download_audio_files(self.df, self.save_path, self.verbose, csv_filename)
        
        if self.verbose:
            print('\n====================== TRAINING THE MODEL ======================\n')
        
        # Train model on top of BirdNET with pseudo-labeled data and save custom classifier
        history = self.trainModel(custom_classifier_path=model_output_path) # trainModel() function is inherited from BirdNET() class
        filename, _ = os.path.splitext(model_output_path)
        
        # Update model and label file paths
        cfg.MODEL_PATH = model_output_path
        cfg.LABELS_FILE = f"{filename}_Labels.txt"
        
        return history
        

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
                    # _extractFeatures() function is inherited from BirdNET() class
                    for df_embeddings_temp in pool.map(partial(self._extractFeatures), (row for _, row in self.df.iterrows())):
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

        if csv_filename is not None:
            if self.save_path is not None:
                # Format save_path into Path
                save_path = Path(self.save_path)
                # Save and append dataframe 
                csv_filename = save_path / csv_filename
            
            self.df.to_csv(csv_filename, sep=";", encoding='utf-8', index=False, index_label='id')
        
        return self.df
    

    def pseudo_labeling(self, dataset=None, display=False, csv_filename=None):
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
            csv_filename (str, optional): The name of the csv file to save the pseudo-labeled data. Default is None.

        Returns:
            pd.DataFrame: DataFrame containing all the pseudo-labeled data.
        """

        if self.verbose:
            print(f'\n====================== PSEUDO-LABELING DATA WITH {cfg.METHOD} ======================\n')    
        
        if dataset is not None:
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
                from cluster import prepare_embeddings, estimate_parameters
                
                X = prepare_embeddings(df_single_categories, scaler=cfg.SCALER, embeddings=cfg.FEATURES)
                #-------------------------------------------------------
        
                if cfg.MIN_PTS == 'auto':    
                    # Automatic estimation of the clustering algorithm parameters
                    #--------------------------------------------------------------------
                    range_n_components = [2,3,4,5,6,7,8,9,10,11]
                    dbscan_parameters = []
                    
                    # Multi-CPU process
                    #-------------------                    
                    with ProcessPoolExecutor(max_workers=cfg.CPU_THREADS-1) as executor:
                        futures = [executor.submit(estimate_parameters, X, n_components) for n_components in range_n_components]
                        for future in tqdm(futures, desc=f"Estimate {cfg.METHOD} parameters"):
                            # Store or process the results
                            results = future.result()
                            if not np.isnan(results[0]):  
                                dbscan_parameters.append(results)  
                        
                    # Find the sublist with the highest DBCV score
                    best_score = max(dbscan_parameters, key=lambda x: x[0])
                    dbcv = best_score[0]
                    n_components = best_score[1]
                    cluster = best_score[2]
                    
                else:
                    dbcv, _, cluster = estimate_parameters(X)
                    n_components = None
                    
                if self.verbose:                
                    # Get the predicted labels and the number of clusters 
                    clusters = len(np.unique(cluster.labels_[cluster.labels_ >= 0])) # remove noisy samples
                    percentage = int(round(np.sum(cluster.labels_ >= 0) / X.shape[0], 2) * 100)             
                    print(f'\nClustering {percentage}% of the data: Found {clusters} number of soundtypes with n_components={n_components}.')
                    print(f'Density-Based Clustering Validation: {dbcv}')

                # Add the label found with the clustering
                #-------------------------------------------------------
                # Add the cluster label into the label's column of the dataframe
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
                        # Set by default to 1 the pseudo_label of the biggest cluster
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
                    
                    # # If there is only one category we can analyze the repertoire of the species
                    # if len(df_embeddings.categories.unique()) == 1:
                    #     # Add the cluster number to the categories to differentiate them
                    #     df_cluster['categories'] = df_cluster.apply(
                    #         lambda row: f"{row['categories']}{row['cluster_number']}", axis=1)   

            if display:
                from display import interactive_display
                interactive_display(self.df, output_filename=categories, thumbnails=True, noise=False)
                
            # Increment
            count += 1
                
        # Default name of the csv file with the cluster
        #-------------------------------------------------------------------------
        if csv_filename is not None:
            if self.save_path is not None:
                # Format save_path into Path
                save_path = Path(self.save_path)
                # Save and append dataframe 
                csv_filename = save_path / csv_filename
            
            self.df.to_csv(csv_filename, sep=";", encoding='utf-8', index=False, index_label='id')

        return self.df
           

if __name__ == "__main__":
    
    # Create a Xeno-Canto network
    xcnet = XCNet()
    
    # Build Xeno-Canto dataset with Xeno-Canto parameters from the config file (config.py)
    xcnet.dataset() 
    # Train model on top of BirdNET with Xeno-Canto dataset and save custom classifier to disk
    xcnet.training(model_output_path="checkpoints/custom/XCNet_001.tflite", csv_filename='train_data.csv')
    # Extract embeddings from audio files in 'test_data' folder with XCNet
    xcnet.embeddings(dataset='test_data', csv_filename='embeddings.csv')    
    # Pseudo-label data and display clusters
    xcnet.pseudo_labeling(display=True, csv_filename='cluster.csv') 
    
    
    # # Create a Xeno-Canto network with custom classifier
    # xcnet = XCNet(custom_classifier_path="checkpoints/custom/XCNet.tflite")
    
    
    

    
