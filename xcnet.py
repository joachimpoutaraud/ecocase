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
                save_path = "train_data",
                ):

        self.params_xc = params_xc
        self.species_xc = species_xc
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
        
    def train(self, dataset=None, model_output_path=cfg.CUSTOM_CLASSIFIER, csv_filename=None):
        """
        Train custom classifier on top of BirdNET model with pseudo-labeled data requested from the Xeno-Canto database.
        By default the model is train using the training dataframe with pseudo-labeled data requested from Xeno-Canto.
        It is however also possible to use a training dataset of labeled audio files, organized in folders by species.
        
        Args:
            dataset (str, optional): Path to dataset. This could be the path to a pandas DataFrame or audio recordings folder. Defaults to None (which corresponds to self.df). 
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

        # Build pseudo-labeled dataset if no training dataset is provided        
        if cfg.TRAIN_DATA_PATH is None:
            
            # Check if dataset contains already pseudo labels or not
            if 'pseudo_label' not in self.df.columns:                
                # Build a pseudo-labeled dataset with density-based clustering method       
                self.df = self.embeddings() # extract embeddings from BirdNET model first 
                # import pandas as pd 
                # self.df = pd.read_csv('train_data/embeddings.csv', sep=";")            
                # Find cluster and assign pseudo-labels
                self.df = self.pseudo_label(csv_filename=csv_filename)
            
            # Keep only data pseudo-labeled as signal
            self.df = self.df[self.df.pseudo_label > 0]
        
        if self.verbose:
            print('====================== TRAINING THE MODEL ======================\n')
        
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
    

    def pseudo_label(self, dataset=None, embedder=None, train_df=None, display=False, csv_filename=None):
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
                from cluster import estimate_parameters
                X = df_single_categories.loc[:, df_single_categories.columns.str.startswith("emb")].to_numpy() 

                if embedder is not None:

                    # Prepare the embeddings of the training dataset
                    df_train = pd.read_csv(train_df, sep=";")
                    X1 = df_train.loc[:, df_train.columns.str.startswith("emb")].to_numpy()
                    # Select landmarks indexes
                    landmark_idx = list(np.random.choice(range(X1.shape[0]), int(X1.shape[0]/100), replace=False))

                    # Add the landmark points to x2 for training.
                    X2 = np.concatenate((X, X1[landmark_idx]))

                    # Make our landmarks vector, which is nan where we have no landmark information.
                    embedder.transform(X) 

                    landmarks = np.stack(
                        [np.array([np.nan, np.nan])]*X.shape[0] + list(
                            embedder.transform(
                                X1[landmark_idx]
                            )
                        )
                    )

                    # Set landmark loss weight and continue training our Parametric UMAP model.
                    embedder.landmark_loss_weight = 0.01
                    embedder.fit(X2, landmark_positions=landmarks)
                    X = embedder.transform(X)
              
                config = cfg.get_config()

                if cfg.MIN_PTS == 'auto' and len(df_single_categories) > 500:    
                    # Automatic estimation of the clustering algorithm parameters
                    #--------------------------------------------------------------------
                    if self.verbose:
                        print(f'Automatic estimation of min_samples parameter...')
                    
                    dbscan_parameters = []
                    best_score = None
                    n_components = 2
                    
                    # Multi-CPU process
                    #-------------------                    
                    with ProcessPoolExecutor(max_workers=cfg.CPU_THREADS-1) as executor:
                        while True:
                            future = executor.submit(estimate_parameters, X, config, n_components, cfg.METRIC)
                            results = future.result()
                            if not np.isnan(results[0]):  
                                dbscan_parameters.append(results)
                                
                                # Check if the current score is lower than the previous best score
                                if best_score is not None and results[0] < best_score[0]:
                                    break
                                
                                # Update the best score
                                best_score = results
                            # Increment the number of components
                            n_components += 1
                        
                    # Use the best score found
                    dbcv = best_score[0]
                    n_components = best_score[1]
                    cluster = best_score[2] 
                    eps = best_score[3]
                    min_samples = best_score[4]
                
                else:
                    dbcv, n_components, cluster, eps, min_samples = estimate_parameters(X, config, metric=cfg.METRIC)
                    
                if self.verbose:                
                    # Get the predicted labels and the number of clusters 
                    clusters = len(np.unique(cluster.labels_[cluster.labels_ >= 0])) # remove noisy samples
                    percentage = int(round(np.sum(cluster.labels_ >= 0) / X.shape[0], 2) * 100)  
                    print(f'Data clustering percentage (without noise): {percentage}%')           
                    print(f'Number of clusters found: {clusters} (n_components={n_components}, min_samples={min_samples}, eps={eps})')
                    print(f'Density-Based Clustering Validation: {dbcv}\n')

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
                    interactive_display(self.df, output_filename=categories, save_path=save_path, thumbnails=True, noise=False)
                
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
    xcnet = XCNet(save_path='train_data') # create 'train_data' folder to save data
    # Build Xeno-Canto dataset with Xeno-Canto parameters from the config file (config.py)
    xcnet.dataset(csv_filename="xc_dataset.csv") 
    # xcnet.df = pd.read_csv("train_data/train_dataset.csv", sep=";")

    # We use specific clustering settings for building pseudo-labeled dataset
    config = cfg.get_config()
    config["KEEP"] = 'BIGGEST'
    if len(xcnet.df.categories.unique()) == 1: 
        config["KEEP"] = 'ALL' # keep all clusters for repertoire analysis on an unique species
    cfg.set_config(config) # restore config in case we're on Windows to be thread save
    
    # Train model on top of BirdNET with Xeno-Canto dataset and save to disk
    embedder = xcnet.train(model_output_path="checkpoints/custom/XCNet.tflite", csv_filename='train_dataset.csv')


    # xcnet = XCNet(save_path='test_data')
    # # Extract embeddings from audio files in 'test_data' folder with custom XCNet
    # # xcnet.embeddings(dataset='test_data', csv_filename='embeddings.csv') 

    # # We use specific clustering settings for extracting the clusters
    # config = cfg.get_config()
    # config["KEEP"] = 'ALL'
    # cfg.set_config(config) # restore config in case we're on Windows to be thread save
   
    # # Pseudo-label data and display clusters
    # dataset = pd.read_csv("test_data/embeddings.csv", sep=";")
    # xcnet.pseudo_label(dataset=dataset, embedder=embedder, train_df='train_data/train_dataset.csv', display=True, csv_filename='cluster.csv')
    # # xcnet.pseudo_label(embedder=embedder, train_df='train_data/train_dataset.csv', display=True, csv_filename='cluster.csv')