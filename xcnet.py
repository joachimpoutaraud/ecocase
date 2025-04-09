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
        
    def train(self, dataset=None, csv_filename=None):
        """
        Train custom classifier on top of BirdNET model with pseudo-labeled data requested from the Xeno-Canto database.
        By default the model is train using the training dataframe with pseudo-labeled data requested from Xeno-Canto.
        It is however also possible to use a training dataset of labeled audio files, organized in folders by species.
        
        Args:
            dataset (str, optional): Path to dataset. This could be the path to a pandas DataFrame or audio recordings folder. Defaults to None (which corresponds to self.df). 
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
                # Extract embeddings from BirdNET model       
                self.df = self.embeddings()         
                # Find cluster and assign pseudo-labels
                self.df = self.pseudo_label(csv_filename=csv_filename)
            
            # Keep only data pseudo-labeled as signal
            self.df = self.df[self.df.pseudo_label > 0]
        
        if self.verbose:
            print('====================== TRAINING THE MODEL ======================\n')
        
        # Train model on top of BirdNET with pseudo-labeled data and save custom classifier
        embedder = self.trainModel(custom_classifier_path=self.save_path) # trainModel() function is inherited from BirdNET() class        
        # Update model path
        cfg.MODEL_PATH = self.save_path
        
        return embedder
        

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

        if self.verbose:
            print(f'Number of audio chunks extracted: {len(self.df)}')

        if csv_filename is not None:
            if self.save_path is not None:
                # Format save_path into Path
                save_path = Path(self.save_path)
                # Save and append dataframe 
                csv_filename = save_path / csv_filename
            
            self.df.to_csv(csv_filename, sep=";", encoding='utf-8', index=False, index_label='id')
        
        return self.df
    

    def pseudo_label(self, dataset=None, embedder=None, train_df=None, display=False, thumbnails=True, noise=False, csv_filename=None):
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
                    df_train = df_train[df_train.pseudo_label > 0]
                    X1 = df_train.loc[:, df_train.columns.str.startswith("emb")].to_numpy()

                    from tensorflow import keras
                    def kullback_leibler_divergence_loss(y_true, y_pred):
                        # Reshape y_true and y_pred to ensure they have the same rank.
                        y_true = keras.backend.flatten(y_true)
                        y_pred = keras.backend.flatten(y_pred)
                        # Computes the cosine similarity between labels and predictions.
                        # Relu activation ensures non-negative loss values.
                        return keras.activations.relu(keras.losses.kullback_leibler_divergence(y_true, y_pred)) # cosine_similarity?
                    
                    embedder.landmark_loss_fn = kullback_leibler_divergence_loss

                    # Select 1 percent of the landmarks indexes randomly
                    landmark_idx = list(np.random.choice(range(X1.shape[0]), int(X1.shape[0] * 0.01), replace=False))
                    # Add the landmark points to X2 for training.
                    X2 = np.concatenate((X, X1[landmark_idx]))
                    # Make our landmarks vector, which is nan where we have no landmark information.
                    landmarks = np.stack([np.array([np.nan, np.nan])] * X.shape[0] + list(embedder.transform(X1[landmark_idx])))

                    # Set landmark loss weight and continue training our Parametric UMAP model.
                    embedder.landmark_loss_weight = 0.01
                    embedder.fit(X2, landmark_positions=landmarks)
                    # Transform the data with the trained model
                    X = embedder.transform(X)
              
                config = cfg.get_config()

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
            if self.save_path is not None:
                # Format save_path into Path
                save_path = Path(self.save_path)
                # Save and append dataframe 
                csv_filename = save_path / csv_filename
            
            self.df.to_csv(csv_filename, sep=";", encoding='utf-8', index=False, index_label='id')

        return self.df
           

if __name__ == "__main__":
    
    ### TRAIN
    xcnet = XCNet(save_path='train_data') # create Xeno-Canto network with 'train_data' folder to save data

    # We use specific clustering settings for building pseudo-labeled dataset
    config = cfg.get_config()
    config["KEEP"] = 'BIGGEST'
    cfg.set_config(config) # restore config in case we're on Windows to be thread save

    # # Build Xeno-Canto dataset with Xeno-Canto parameters from the config file (config.py)
    # xcnet.dataset(csv_filename="xc_dataset.csv")     
    # # Extract embeddings from BirdNET model
    # xcnet.embeddings(csv_filename='embeddings.csv')   
    # # Find cluster and assign pseudo-labels
    # xcnet.pseudo_label(csv_filename='pseudo_label.csv') # dataset="train_data/embeddings.csv", display=True, thumbnails=True, noise=False

    # Train ParametricUMAP model with Xeno-Canto dataset 
    embedder = xcnet.train(dataset="train_data/pseudo_label.csv")


    ### TEST
    xcnet = XCNet(save_path='test_data')

    # We use specific clustering settings for extracting the clusters
    config = cfg.get_config()
    config["KEEP"] = 'ALL'
    cfg.set_config(config) # restore config in case we're on Windows to be thread save

    # Extract embeddings from audio files in 'test_data' folder with custom XCNet
    # xcnet.embeddings(dataset='test_data', csv_filename='embeddings.csv')    
    # Pseudo-label data and display clusters
    xcnet.pseudo_label(embedder=embedder, train_df='train_data/pseudo_label.csv', display=True, csv_filename='pseudo_label.csv', dataset="test_data/embeddings.csv")
    
    
    
    # xcnet.pseudo_label(dataset="test_data/embeddings.csv", csv_filename='pseudo_label.csv', display=True)
    
    
    




    
    # if len(xcnet.df.categories.unique()) == 1: 
    #     config["KEEP"] = 'ALL' # keep all clusters for repertoire analysis on an unique species
    
    # from umap.parametric_umap import load_ParametricUMAP
    # from umap.parametric_umap import UMAPModel
    # embedder = load_ParametricUMAP('train_data')
    # import keras
    # keras_fit_kwargs = {"callbacks": [
    #     keras.callbacks.EarlyStopping(
    #         monitor='loss',
    #         min_delta=10**-2,
    #         patience=10,
    #         verbose=1,
    #     )
    # ]}
    # embedder.keras_fit_kwargs = keras_fit_kwargs
    # from tensorflow.keras.models import load_model
    # from keras.saving import register_keras_serializable
    # load_model("train_data/parametric_model.keras", custom_objects={"UMAPModel": UMAPModel})