import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

import umap
import hdbscan
from hdbscan.validity import validity_index
from kneed import KneeLocator

import config as cfg

import warnings
warnings.filterwarnings(action="ignore")


def prepare_embeddings(df_embeddings,
                       scaler="STANDARDSCALER",
                       embeddings=["emb"]):
    
    """
    Prepare the embeddings before clustering.

    Parameters
    ----------
    df_embeddings: pandas dataframe
        the dataframe should contain the embeddings 
    scaler: string, optional {"STANDARDSCALER", "ROBUSTSCALER", "MINMAXSCALER"}
        Select the type of scaler uses to normalize the embeddings.
        The default is "STANDARDSCALER".
    embeddings: list of embeddings extracted from BirdNET, optional
        List of embeddings will be used for the clustering. The name of the embeddings
        should be the name of a column in the dataframe. In case of "emb", "emb" means that all the embxx will be used.
        The default is ["emb"].

    Returns
    -------
    X : pandas dataframe
        the dataframe with the normalized embeddings 
    """

    # Select the scaler
    #----------------------------------------------
    if scaler == "STANDARDSCALER":
        scaler = StandardScaler() 
    elif scaler == "ROBUSTSCALER":
        scaler = RobustScaler()
    elif scaler == "MINMAXSCALER":
        scaler = MinMaxScaler() 
    else:
        scaler = StandardScaler()
        print ("*** WARNING *** the scaler {} does not exist. StandarScaler was choosen".format(scaler))
    
    X = pd.DataFrame()
    e = embeddings.copy() # copy the list otherwise the suppression of a feature is kept
    
    # Normalize the shapes
    #----------------------------------------------
    if "emb" in e:
        # create a vector with X in order to scale all embeddings together
        X = df_embeddings.loc[:, df_embeddings.columns.str.startswith("emb")]
        X_vect = X.to_numpy()
        X_shape = X_vect.shape
        X_vect = X_vect.reshape(X_vect.size, -1)
        X_vect = scaler.fit_transform(X_vect)
        X = X_vect.reshape(X_shape)
        # remove "emb" from the list
        e.remove('emb')
    
    X2 = pd.DataFrame()
    
    # Normalize the other features
    #-------------------------------------------------------
    # Test if the features list is not null
    if len(e) > 0 :
        # Add other features like frequency centroid
        X2 = df_embeddings[e]
        # Scale data
        X2 = scaler.fit_transform(X2)

    # Concatenate the features after normalization
    #-------------------------------------------------------
    if len(X) == len(X2):
        # Create a matrix with all features after rescaling
        X = np.concatenate((X, X2), axis=1)
    elif len(X2) > len(X):
        X2 = X
    
    return X


def estimate_parameters(X, n_components=2):
    
    if cfg.MIN_PTS == 'auto':      
        # Reduce dimensionality of the latent space
        embedding = umap.UMAP(n_components=n_components, random_state=cfg.RANDOM_SEED, densmap=True).fit_transform(X)
        # Determine the parameters for DBSCAN (Sander et al., 1998)
        # embedding = X
        min_points = embedding.shape[1] * 2 - 1 
        
    else:
        embedding = umap.UMAP(n_components=2, min_dist=0.0, n_neighbors=30, random_state=cfg.RANDOM_SEED, densmap=True).fit_transform(X)
        
        if cfg.MIN_PTS > 1:
            min_points = round(cfg.MIN_PTS)
        else:
            min_points = 2
    
    if cfg.EPS == 'auto':
        # Calculate the average distance between each point in the data set and
        # its N nearest neighbors (N corresponds to min_points).
        neighbors = NearestNeighbors(n_neighbors=min_points)
        neighbors_fit = neighbors.fit(embedding)
        distances, indices = neighbors_fit.kneighbors(embedding)

        # Sort distance values by ascending value and plot
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1] 

        # Find the knee (curvature inflexion point)
        kneedle = KneeLocator(
            x=np.arange(0, len(distances), 1),
            y=distances,
            interp_method="polynomial",
            curve="convex",
            direction="increasing"
        )

        # First find the maximum distance that corresponds to 95% of observations
        try:
            eps = float(kneedle.knee_y) 
        except:
            eps = 0.1 
        
        if eps == 0:
            eps = float(distances.max())  
            
    else:
        eps = cfg.EPS

    # Perform clustering from embedding array               
    if cfg.METHOD == "DBSCAN":
        cluster = DBSCAN(eps=eps, min_samples=min_points).fit(embedding)
        
    elif cfg.METHOD == "HDBSCAN":
        cluster = hdbscan.HDBSCAN(
            # min_samples=100,
            min_cluster_size=min_points,
            min_samples=round(min_points / 2),
            cluster_selection_epsilon=float(eps),
        ).fit(embedding)
    
    # Evaluate clusters with DBCV score
    dbcv = validity_index(X.astype(np.float64), cluster.labels_, metric='euclidean')
    # Check if dbcv score is not a number
    if math.isnan(dbcv):
        dbcv = 0
    
    return dbcv, n_components, cluster