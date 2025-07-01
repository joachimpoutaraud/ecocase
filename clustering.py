import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import math
import numpy as np
from multiprocessing.pool import Pool

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import hdbscan
from hdbscan.validity import validity_index
from kneed import KneeLocator

from birdnet_analyzer import config as cfg

import warnings
warnings.filterwarnings(action="ignore")


def cluster(df, config, verbose=False):

    # Set configuration parameters
    cfg.set_config(config) 

    # Check if the number of ROIs is higher than 2
    if len(df) < 3: 
        df["prediction"] = -1 # noise
        df["pseudo_label"] = 0 # noise
        
        if verbose:
            print("Only {} ROIs. It requires at least 3 ROIs to perform clustering".format(len(df))) 
    else:
        if verbose:
            print(f"Clustering '{df['categories'][0]}' species with {cfg.METHOD}: {len(df)} ROIs\n")

        # Prepare the embeddings of the categories
        X = df.loc[:, df.columns.str.startswith("emb")].to_numpy()

        if cfg.MIN_PTS == 'auto':
            # Automatic estimation of the clustering algorithm parameters
            min_samples = 3
            dbscan_parameters = []
            best_score = None
            
            # Multi-CPU processsing to estimate the parameters                  
            with Pool(cfg.CPU_THREADS - 1) as pool:
                while True:
                    params = pool.apply(estimate_parameters, args=(X, min_samples, cfg.METRIC))
                    if not np.isnan(params[0]):
                        dbscan_parameters.append(params)
                    # Check if the current score is lower than the previous best score
                    if best_score is not None and params[0] <= best_score[0]:
                        break
                    # Update the best score
                    best_score = params
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

            dbcv, cluster, eps, min_samples = estimate_parameters(X, min_samples, cfg.METRIC)

        # Add the label found with the clustering
        df["prediction"] = cluster.labels_.reshape(-1, 1)

        # Add the automatic label (SIGNAL = 1 or NOISE = 0) into the pseudo_label's column of the dataframe
        # Test if we want to consider only the biggest or all clusters that are not noise (-1) to be signal
        if cfg.KEEP == 'BIGGEST':
            # Set pseudo_label of noise (cluster ID = -1) to 0 
            df.loc[df["prediction"] < 0, "pseudo_label"] = int(0)
            # Set pseudo_label of the biggest cluster to 1
            biggest_cluster_ID = df.loc[df["prediction"] >= 0]["prediction"].value_counts().idxmax()
            df.loc[df["prediction"] == biggest_cluster_ID, "pseudo_label"] = int(1)

            # # Find the cluster (excluding noise and biggest) that is farthest from the biggest cluster
            # # Compute centroids of each cluster (excluding noise)
            # cluster_ids = df["prediction"].unique()
            # cluster_ids = [cid for cid in cluster_ids if cid >= 0 and cid != biggest_cluster_ID]
                
            # # Get the centroid of the biggest cluster
            # biggest_centroid = X[df["prediction"].values.flatten() == biggest_cluster_ID].mean(axis=0)
            # max_dist = -1
            # farthest_cluster_id = None
            # for cid in cluster_ids:
            #     cluster_centroid = X[df["prediction"].values.flatten() == cid].mean(axis=0)
            #     dist = np.linalg.norm(biggest_centroid - cluster_centroid)
            #     if dist > max_dist:
            #         max_dist = dist
            #         farthest_cluster_id = cid
            # # Label the farthest cluster with -1 in pseudo_label to use it as non-event class
            # if farthest_cluster_id is not None:
            #     df.loc[df["prediction"] == farthest_cluster_id, "pseudo_label"] = int(-1)

        elif cfg.KEEP == 'ALL':                
            # Set pseudo_label of noise (cluster ID = -1) to 0 
            df.loc[df["prediction"] < 0, "pseudo_label"] = int(0)
            # Set pseudo_label of signal (cluster ID >= 0) to 1 
            df.loc[df["prediction"] >= 0, "pseudo_label"] = int(1)

        if verbose:                
            # Get the predicted labels and the number of clusters (excluding noise)
            clusters = len(np.unique(cluster.labels_[cluster.labels_ >= 0])) 
            noisy_percentage = round(np.sum(cluster.labels_ == -1) / X.shape[0], 2) * 100          
            print(f'Number of clusters found (excluding noise): {clusters} (min_samples={min_samples}, eps={eps})')
            if cfg.KEEP == 'BIGGEST':
                print(f'Number of ROIs found in the biggest cluster: {df[df["pseudo_label"] == 1].shape[0]}')
            print(f'Percentage of noise/outliers data: {noisy_percentage}%')
            print(f'Density-Based Clustering Validation: {round(dbcv,4)}\n')

        return df


def estimate_parameters(X, min_samples, metric='cosine'):
                
    # Remove NaN values from the dataset
    embedding = np.nan_to_num(X)
    
    if cfg.EPS == 'auto':
        # Calculate the average distance between each point in the data set and
        # its N nearest neighbors (N corresponds to min_samples).
        neighbors = NearestNeighbors(n_neighbors=min_samples, metric=metric)            
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

    # Compute pairwise distance to avoid errors with cosine metric
    # https://github.com/scikit-learn-contrib/hdbscan/issues/69
    from sklearn.metrics import pairwise_distances
    precomputed_distances = pairwise_distances(embedding, metric=metric)

    # Perform clustering from embedding array               
    if cfg.METHOD == "DBSCAN":                
        cluster = DBSCAN(
            min_samples=min_samples,
            eps=float(eps), 
            metric='precomputed',
            ).fit(precomputed_distances.astype('float64'))
        
    elif cfg.METHOD == "HDBSCAN":
        cluster = hdbscan.HDBSCAN(
            min_cluster_size=min_samples,
            cluster_selection_epsilon=float(eps),
            metric='precomputed',
            ).fit(precomputed_distances.astype('float64'))
    
    # Evaluate default BirdNET embeddings with DBCV score
    dbcv = validity_index(embedding.astype('float64'), cluster.labels_, metric=metric)

    # Check if DBCV score is not a number
    if math.isnan(dbcv):
        dbcv = 0.0
         
    return dbcv, cluster, eps, min_samples