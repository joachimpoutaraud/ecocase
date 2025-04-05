import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import math
import numpy as np

import umap
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import hdbscan
from hdbscan.validity import validity_index
from kneed import KneeLocator

import config as cfg

import warnings
warnings.filterwarnings(action="ignore")


def estimate_parameters(X, config, n_components=2, metric='cosine'):
    
    # Restore config in case we're on Windows to be thread save
    cfg.set_config(config)
    
    # Fit the UMAP model and transform data
    reducer = umap.UMAP(n_components=n_components, metric=metric, random_state=cfg.RANDOM_SEED)
    embedding = reducer.fit_transform(X)
    
    if cfg.MIN_PTS == 'auto':      
        # Estimate min_points parameter (Sander et al., 1998)
        min_points = embedding.shape[1] * 2 - 1
 
    else:       
        if cfg.MIN_PTS > 1:
            min_points = round(cfg.MIN_PTS)
        else:
            min_points = 2
    
    if cfg.EPS == 'auto':
        # Calculate the average distance between each point in the data set and
        # its N nearest neighbors (N corresponds to min_points).
        neighbors = NearestNeighbors(n_neighbors=min_points, metric=metric)
            
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

    # Compute pairwise distance to avoid errors
    # https://github.com/scikit-learn-contrib/hdbscan/issues/69
    from sklearn.metrics import pairwise_distances
    precomputed_distances = pairwise_distances(embedding, metric=metric)

    # Perform clustering from embedding array               
    if cfg.METHOD == "DBSCAN":                
        cluster = DBSCAN(
            min_samples=min_points,
            eps=eps, 
            metric='precomputed',
            ).fit(precomputed_distances.astype('float64'))
        
    elif cfg.METHOD == "HDBSCAN":
        cluster = hdbscan.HDBSCAN(
            min_cluster_size=min_points,
            min_samples=min_points,
            cluster_selection_epsilon=float(eps),
            metric='precomputed',
            ).fit(precomputed_distances.astype('float64'))
    
    # Evaluate default BirdNET embeddings with DBCV score
    dbcv = validity_index(X.astype('float64'), cluster.labels_, metric=metric)

    # Check if DBCV score is not a number
    if math.isnan(dbcv):
        dbcv = 0.0
         
    return dbcv, n_components, cluster, eps, min_points