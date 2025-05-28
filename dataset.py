import os
import shutil
import pandas as pd
import numpy as np

import json
import tqdm
import time, datetime
import urllib.request
from pathlib import Path
from functools import partial
from multiprocessing.pool import Pool

from cluster import estimate_parameters

from birdnet_analyzer import audio
from birdnet_analyzer import model
from birdnet_analyzer import config as cfg

import warnings
warnings.filterwarnings(action="ignore")


class XenoCanto():
    
    """
    Query and build multi-species dataset from Xeno-Canto database.
    Xeno-Canto metadata is grouped and stored in a pandas Dataframe.

    Args:
        params_xc (list): List of Xeno-Canto searching parameters to perform the query.
            The main searching parameters include:
            - grp : birds
            - q   : quality
            - cnt : country
            - len : length
            - area : continent (e.g., europe, africa, america, asia)
            See more details here: https://www.xeno-canto.org/help/search
        species_xc (list): List of Xeno-Canto scientific species (e.g., 'Columba palumbus') in the config file.
            If None, species will only be requested according to the Xeno-Canto query parameters.
        num_files (int, optional): Number of audio files for each species. Default is None.
        format_time (bool, optional): Time in Xeno-Canto is not always present or correctly formatted.
            If True, time will be correctly formatted to DateTime format. 
            Rows with unprocessable time data are dropped. Default is False.
        format_date (bool, optional): Date in Xeno-Canto is not always present or correctly formatted.
            If True, rows with incorrectly formatted date data are dropped. Default is False.
        verbose (bool, optional): If True, prints messages during the execution of the function. Default is True.
        save_path (str, optional): Path to the directory where the data will be saved. Default is None.
        overwrite (bool, optional): If True, overwrites the existing directory. Default is True.
 
    Returns:
        pandas.DataFrame: Dataframe containing all the recordings metadata matching the Xeno-Canto searching parameters.
    """

    def __init__(
                self,
                params_xc = cfg.PARAMS_XC,
                species_xc = cfg.SPECIES_XC,
                num_files = cfg.NUM_FILES,
                format_time = False,
                format_date = False,
                verbose = True,
                save_path = 'train_data',
                overwrite = True,
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


    def data(self, csv_filename=None):
        """
        Query Xeno-Canto database and build a dataset from the metadata.
        The dataset is a pandas dataframe containing all the recordings metadata matching the Xeno-Canto searching parameters.

        Args:
            csv_filename (str, optional): Name of the CSV file to save the dataset. Default is None.
                If None, the dataset will not be saved.

        Returns:
            pandas.DataFrame: Dataframe containing all the recordings metadata matching the Xeno-Canto searching parameters.
        """
        
        if self.verbose:
            print('\n========================= QUERY DATA =========================\n')
            
        if self.species_xc is None:
            self.df = self.query()     
        else:        
            # Separate the genus and the species from the scientific name
            gen, sp = [], []
            
            for name in self.species_xc:
                gen.append(name.rpartition(' ')[0])
                sp.append(name.rpartition(' ')[2])

            # Add the genus and the species as query parameters
            df_query = pd.DataFrame() # create empty dataframe
            df_query[0] = gen
            df_query[1] = sp

            # Add the other parameters
            if self.params_xc is not None:
                idx = 2
                for param in self.params_xc:
                    df_query[idx] = param
                    idx += 1
            
            # Create empty dataframe
            self.df = pd.DataFrame() 
                     
            for _, row in df_query.iterrows():
                # Convert parameters to list
                self.params_xc = row.tolist()
                # Request data from Xeno-Canto
                self.df = pd.concat([self.df, self.query()]) 
                
        if self.verbose:            
            if len(self.df) > 0: 
            # Test if the dataframe contains files to download
                print('\n====================== XENO-CANTO DATASET ======================\n')
                print('Composition of the dataset:')
                print('   - Number of audio files: %2.0f' % len(self.df))
                print('   - Number of species: %2.0f' % len(self.df['categories'].unique()))
                print('   - Unique species: {}\n'.format(sorted(self.df['categories'].unique())))
            else:
                print('*** WARNING ***\nThe dataframe is empty.')   
                print("No audio recording was found with the parameters:\n{}".format(self.params_xc))
                return
                
        # Test if recordings are found
        if len(self.df) > 0:
            # Rearrange index to be sure to have unique and increasing index
            self.df.reset_index(drop=True, inplace=True)
            
        if csv_filename is not None:            
            if self.save_path is not None:
                # Format save_path into Path
                save_path = Path(self.save_path)
                # Save and append dataframe 
                csv_filename = save_path / csv_filename
            # Save dataframe
            self.df.to_csv(csv_filename, sep=";", encoding='utf-8', index=False, index_label='id') 

        return self.df
    

    def query(self):
        """ 
        Query Xeno-Canto database and build a dataset from the metadata.

        Returns:        
            pandas.DataFrame: Dataframe containing all the recordings metadata matching the Xeno-Canto searching parameters.
        """
        
        # Create empty dataframe
        df = pd.DataFrame() 
        
        numPages = 1
        page = 1
        
        while page < numPages + 1:
            try:
                params_xc = '%20'.join(self.params_xc)
                url = 'https://www.xeno-canto.org/api/2/recordings?query={0}&page={1}'.format(params_xc, page)        
                jsonPage = urllib.request.urlopen(url) 
            except UnicodeEncodeError:
                params_xc = urllib.parse.quote(' '.join(self.params_xc), safe=':/')
                url = 'https://www.xeno-canto.org/api/2/recordings?query={0}&page={1}'.format(params_xc, page)
                jsonPage = urllib.request.urlopen(url)
                
            if self.verbose:
                print("Loading page "+str(page)+"...")
                print(url)
             
            jsondata = json.loads(jsonPage.read().decode('utf-8'))
            # check number of pages
            numPages = jsondata['numPages']
            # Concatenate pandas dataframe of records & convert to .csv file            
            df = pd.concat([df, pd.DataFrame(jsondata['recordings'])], ignore_index=True) 
            # increment the current page
            page += 1
                    
        # test if the dataset is not empty
        if len(df) > 0:
            # convert latitude and longitude coordinates into float
            try:
                df['lat'] = df['lat'].astype(float)
                df['lng'] = df['lng'].astype(float)
            except ValueError:
                pass
            
            # rearrange index to be sure to have unique and increasing index
            df.reset_index(drop=True, inplace=True)
            
            # the format of length is not correct (missing 0 before 0:45 => 00:45)
            # Correct the format of length for length shorten than 9:59 (4 characters) by adding a 0
            df['length'].where(~(df.length.str.len()==4), 
                                        other='0'+ df[df.length.str.len()==4].length, 
                                        inplace=True) 
            
            if self.format_time:
                # rearrange index to be sure to have unique and increasing index
                df.reset_index(drop=True, inplace=True)
                
                # the format of time is not always correct
                # replace . by :
                df['time'].replace(to_replace = '[.]', value=':', regex= True)
                df['time'].replace(to_replace = '[ ] ', value='', regex= True)
                
                # drop rows where there is no valid time information that can be corrected
                df = df[(df.time.str.match('^(0[0-9]|1[0-9]|2[0-3])[:]([0-5][0-9])$')) | 
                                        (df.time.str.match('^([0-9])[:]([0-5][0-9])$'))]
                
                # Correct the format of time when 0 is missing (missing 0 before 0:45 => 00:45) by adding a 0
                df['time'][df.time.str.match('^([0-9])[:]([0-5][0-9])$')] = '0' + df[df.time.str.match('^([0-9])[:]([0-5][0-9])$')].time

                if self.verbose:
                    print("Keeped metadata for", len(df), "files after formating time")
                
            if self.format_date:
                # rearrange index to be sure to have unique and increasing index
                df.reset_index(drop=True, inplace=True)
                # drop rows where there is no valid date information
                df = df[df.date.str.match(r'^(20[0-9][0-9]|19[0-9][0-9])-(0[1-9]|1[0-2])-([1-9]|1[0-9]|2[0-9]|3[0-1])$')]
                
                if self.verbose:
                    print("Keeped metadata for", len(df), "files after formating date")
            
            if (self.format_time == True) and (self.format_date == True) :         
                # add a column with the week number
                df['week'] = pd.to_datetime(df['date']).dt.isocalendar()['week'] # type: ignore
                # add a column with datetime in DateTime format
                df['datetime'] =  pd.to_datetime(df['time']+' '+ df['date'], format="%H:%M %Y-%m-%d")

        if len(df) > 0:    
            if self.num_files is not None:
                if self.num_files > len(df):
                    if self.verbose:
                        print(f"Only {len(df)} audio files are available.")
                else:
                    # Test if the number of files is greater than the maximum number of resquested files for each species 
                    for i in df.en.value_counts()[df.en.value_counts() > self.num_files].index.to_list():
                        rows = df[df.en == i]
                        to_drop = rows.sample(n=len(rows) - self.num_files, random_state=cfg.RANDOM_SEED).index.to_list()
                        df = df.drop(to_drop)
                    # Test if the number of files is smaller than the minimum number of resquested files for each species 
                    for i in df.en.value_counts()[df.en.value_counts() < self.num_files].index.to_list():
                        to_drop = df[df.en == i].index.to_list()
                        df = df.drop(to_drop)
                        
        # Convert length column to seconds
        def convert_to_seconds(length_str):
            length = time.strptime(length_str, '%M:%S')
            length_seconds = int(datetime.timedelta(hours=length.tm_hour, minutes=length.tm_min, seconds=length.tm_sec).total_seconds())
            return length_seconds
        
        df['length'] = df['length'].apply(convert_to_seconds) 
                
        # Rearrange index to be sure to have unique and increasing index
        df.reset_index(drop=True, inplace=True)
        
        # Create a categories column
        df["categories"] = df.gen + " " + df.sp
        # Reorder columns to make 'categories' the second column
        cols = df.columns.tolist()
        cols.insert(1, cols.pop(cols.index('categories')))
        df = df[cols]                

        return df
    
    
    def labeling(self, dataset=None, cache_filename=None):
        """ 
        Label the Xeno-Canto dataset using density-based clustering.

        Args:   
            dataset (str or pd.DataFrame, optional): Path to the dataset CSV file or a pandas DataFrame. Default is None.
            cache_filename (str, optional): Name of the cache file to save the labeled data. Default is None.
                If None, the cache file will not be saved.            
            
        Returns:
            x_train (np.ndarray): The training data embeddings.
            y_train (np.ndarray): The training labels.
        """

        if dataset is not None:
            self.df = pd.read_csv(dataset, sep=";")

        # Create a new dataframe for labeling
        df_labeling = pd.DataFrame()

        unique_categories = self.df['categories'].unique()
        categories = list(sorted(unique_categories))

        # Get valid labels
        valid_labels = [label for label in categories if label.lower() not in cfg.NON_EVENT_CLASSES and not label.startswith("-")]

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

        for cat in categories:
            # Get label vector
            label_vector = np.zeros((len(valid_labels),), dtype="float32")
            cat_labels = cat.split(",")

            for label in cat_labels:
                if label.lower() not in cfg.NON_EVENT_CLASSES and not label.startswith("-"):
                    label_vector[valid_labels.index(label)] = 1
                elif (
                    label.startswith("-") and label[1:] in valid_labels
                ):  # Negative labels need to be contained in the valid labels
                    label_vector[valid_labels.index(label[1:])] = -1

            # Get list of file urls
            file_urls = self.df[self.df['categories'] == cat]['file'].tolist()

            timestamps = []
            file_paths = []
            cat_embeddings = []

            # Load files using thread pool
            with Pool(cfg.CPU_THREADS) as p:
                tasks = []

                for f in file_urls:
                    task = p.apply_async(partial(self.load_xc_file, f=f, label_vector=label_vector, config=cfg.get_config()))
                    tasks.append(task)

                # Wait for tasks to complete and monitor progress with tqdm
                if self.verbose:
                    with tqdm.tqdm(total=len(tasks), desc=f"Loading '{cat}' species", unit="f") as progress_bar:
                        for task in tasks:
                            result = task.get()

                            # Check if the result is not empty
                            if len(result[0]) > 0:
                                cat_embeddings += result[0]
                                timestamps += result[2]
                                file_paths += result[3]

                            # Update progress bar
                            progress_bar.update(1)
                else:
                    for task in tasks:
                        result = task.get()

                        # Check if the result is not empty
                        if len(result[0]) > 0:
                            cat_embeddings += result[0]
                            timestamps += result[2]
                            file_paths += result[3]

            # Format timestamps and embeddings into new dataframe 
            cols = ['start', 'end'] 
            cols += ['emb' + str(idx).zfill(4) for idx in range(1, len(cat_embeddings[0]) + 1)] 
            data = np.concatenate([np.asarray(timestamps), np.asarray(cat_embeddings)], axis=1)

            # Create a dataframe with the data
            df_single_categories = pd.DataFrame(data=data, columns=cols)
            df_single_categories['categories'] = cat
            df_single_categories['file'] = file_paths

            # Find the cluster for each categories separately 
            if len(df_single_categories) < 3: # test if the number of ROIs is higher than 2
                df_single_categories["cluster_number"] = -1 # noise
                df_single_categories["pseudo_label"] = 0 # noise
                
                if self.verbose:
                    print("Only {} ROIs. It requires at least 3 ROIs to perform clustering".format(len(df_single_categories)))
                    
            else:
                config = cfg.get_config()

                # Prepare the embeddings of the categories
                X = df_single_categories.loc[:, df_single_categories.columns.str.startswith("emb")].to_numpy() 

                if self.verbose:
                    print(f"Labeling '{cat}' species with {cfg.METHOD}: {len(df_single_categories)} ROIs\n")

                if cfg.MIN_PTS == 'auto':
                    # Automatic estimation of the clustering algorithm parameters
                    min_samples = 3
                    dbscan_parameters = []
                    best_score = None
                    
                    # Multi-CPU processsing to estimate the parameters                  
                    with Pool(cfg.CPU_THREADS - 1) as pool:
                        while True:
                            params = pool.apply(estimate_parameters, args=(X, config, min_samples, cfg.METRIC))
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

                    dbcv, cluster, eps, min_samples = estimate_parameters(X, config, min_samples, cfg.METRIC)

                if self.verbose:                
                    # Get the predicted labels and the number of clusters (excluding noise)
                    clusters = len(np.unique(cluster.labels_[cluster.labels_ >= 0])) 
                    noisy_percentage = round(np.sum(cluster.labels_ == -1) / X.shape[0], 2) * 100           
                    print(f'Number of clusters found (excluding noise): {clusters} (min_samples={min_samples}, eps={eps})')
                    print(f'Percentage of noise/outliers data: {noisy_percentage}%')
                    print(f'Density-Based Clustering Validation: {round(dbcv,4)}\n')

                # Add the label found with the clustering
                df_single_categories["cluster_number"] = cluster.labels_.reshape(-1, 1)
        
                # Add the automatic label (SIGNAL = 1 or NOISE = 0) into the pseudo_label's column of the dataframe
                # Test if we want to consider only the biggest or all clusters that are not noise (-1) to be signal
                
                if cfg.KEEP == 'BIGGEST':
                    # Set by default to 0 the pseudo_label of all
                    df_single_categories.loc["pseudo_label"] = int(0)

                    # Find the cluster ID of the biggest cluster that is not noise
                    try:
                        # Set to 1 the pseudo_label of the biggest cluster
                        biggest_cluster_ID = df_single_categories.loc[df_single_categories["cluster_number"] >= 0]["cluster_number"].value_counts().idxmax()
                        df_single_categories.loc[df_single_categories["cluster_number"] == biggest_cluster_ID, "pseudo_label"] = int(1)
                    except:
                        # If there is only noise
                        pass
                    
                elif cfg.KEEP == 'ALL':                
                    # Set by to 0 the pseudo_label of the noise (cluster ID = -1)
                    df_single_categories.loc[df_single_categories["cluster_number"] < 0, "pseudo_label"] = int(0)
                    # Set by to 1 the pseudo_label of the signal (cluster ID >= 0)
                    df_single_categories.loc[df_single_categories["cluster_number"] >= 0, "pseudo_label"] = int(1)

                # Concatenate dataFrames
                df_labeling = pd.concat([df_labeling, df_single_categories], ignore_index=True)

        # Keep only data pseudo-labeled as signal
        self.df = df_labeling[df_labeling.pseudo_label == 1]
        # Remove '0' categories if it exists
        self.df = self.df[self.df.categories != 0]
        # Rearrange index to be sure to have unique and increasing index
        self.df.reset_index(drop=True, inplace=True)

        # Get the embeddings
        x_train = self.df.loc[:, self.df.columns.str.startswith("emb")].to_numpy()
        # Perform one-hot encoding on categories
        one_hot_encoded = pd.get_dummies(self.df.categories.astype(str))
        y_train = one_hot_encoded.values

        if cache_filename is not None:
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


    def download(self, dataset=None, csv_filename=None):
        """
        Download audio files from Xeno-Canto.
        
        Args:
            dataset (str or pd.DataFrame, optional): Path to the dataset CSV file or a pandas DataFrame. Default is None.
            csv_filename (str, optional): The name of the csv file to save the ROIs. Default is None.
        """ 
            
        if self.verbose:
            print('\n====================== DOWNLOAD AUDIO FILES ======================\n')
            
        # Check if dataset is a path to a dataframe, audio file or folder
        if type(dataset) == str:
            ext = os.path.splitext(dataset)[1]
            # Check file extension
            if ext == ".csv": 
                import pandas as pd
                self.df = pd.read_csv(dataset, sep=";")        

        # Create folders for each species if needed
        if 'categories' in self.df.columns:
            species_list = [s.replace(' ', '_') for s in self.df.categories.unique()]

            # Create folders for each species
            for species in species_list:
                os.makedirs(os.path.join(self.save_path, species), exist_ok=True)

        # Process rows in parallel
        nb_cpu = os.cpu_count()
        updated_file_paths = []
        
        with tqdm.tqdm(total=len(self.df), desc="Download audio files") as pbar:
            with Pool(processes=nb_cpu - 1) as pool:
                # Prepare arguments for starmap
                rows = [row for _, row in self.df.iterrows()]
                # Use partial to pass save_path if needed, or just row if not
                for output_file in pool.imap(self.write_xc_file, rows):
                    updated_file_paths.append(output_file)
                    pbar.update(1)

        if csv_filename is not None:
            # Format save_path into Path
            save_path = Path(self.save_path)
            # Save and append dataframe 
            csv_filename = save_path / csv_filename
            
            # Update the audio file paths of the dataframe 
            self.df['file'] = updated_file_paths         
            self.df.to_csv(csv_filename, index=False, sep=";")


    def write_xc_file(self, row):
        """
        Write audio file from Xeno-Canto to the specified path.
        Args:
            row (pd.Series): A row from the DataFrame containing the audio file metadata.

        Returns:
            str: The path to the saved audio file.   
        """
        
        import soundfile as sf
        from urllib.parse import urlparse
        
        try:
            # Check if the file is a URL or a local file
            path = urlparse(row['file']).path
            id_number = path.split('/')[1]

            if 'start' in row:
                sig, rate = self.open_xc_file(row['file'], cfg.SAMPLE_RATE, row.start, duration=row.end - row.start)
                output_file = os.path.join(os.path.join(self.save_path, row['categories'].replace(' ', '_')), f"XC{int(id_number)}_{float(row['start'])}_{float(row['end'])}.wav")
            else:
                sig, rate = self.open_xc_file(row['file'], cfg.SAMPLE_RATE)
                output_file = os.path.join(os.path.join(self.save_path, row['categories'].replace(' ', '_')), f"XC{int(id_number)}.wav")

            sf.write(output_file, sig, rate)
            
            # Return updated file path
            return output_file
            
        except Exception as ex:
            print(f"Error: Cannot analyze audio file {row.file}.", flush=True)
            from birdnet_analyzer import utils
            utils.write_error_log(ex)

    
    def load_xc_file(self, f, label_vector, config):
        """
        Load an audio file from Xeno-Canto and extract features.

        Args:
            f: Path to the audio file.
            label_vector: The label vector for the file.
            config: Configuration settings.

        Returns:
            A tuple of (x_train, y_train).
        """

        x_train = []
        y_train = []

        # restore config in case we're on Windows to be thread save
        cfg.set_config(config)

        # Try to load the audio file
        try:
            # Load audio
            sig, rate = self.open_xc_file(
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

        # Get audio timestamps and file paths
        timestamps = []
        file_paths = []
        start = 0
        end = start + cfg.SIG_LENGTH

        for s in range(len(sig_splits)):
            timestamps.append([start, end])
            file_paths.append(f)
            # Advance start and end
            start += cfg.SIG_LENGTH - cfg.SIG_OVERLAP
            end = start + cfg.SIG_LENGTH
        
        # Get feature embeddings
        batch_size = 1  # turns out that batch size 1 is the fastest, probably because of having to resize the model input when the number of samples in a batch changes
        for i in range(0, len(sig_splits), batch_size):
            batch_sig = sig_splits[i : i + batch_size]
            batch_label = [label_vector] * len(batch_sig)
            e = model.embeddings(batch_sig)

            # Add to training data
            x_train.extend(e)
            y_train.extend(batch_label)

        return x_train, y_train, timestamps, file_paths


    def open_xc_file(self, path: str, rate=cfg.SAMPLE_RATE, offset=0.0, duration=None, fmin=None, fmax=None):
        """
        Open an audio file from Xeno-Canto with FFmpeg and the given settings.

        Args:
            path (str): Path or URL to the audio file.
            sample_rate (int): The sample rate at which the file should be processed.
            offset (float): The starting offset.
            duration (float): Maximum duration of the loaded content.
            fmin (float): Minimum frequency for bandpass filtering.
            fmax (float): Maximum frequency for bandpass filtering.

        Returns:
            Returns the audio time series and the sampling rate.
        """

        def open_audio_with_ffmpeg(command, retries=3, delay=30):

            import subprocess
            import time

            for attempt in range(retries):
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out, err = process.communicate()
                
                if process.returncode == 0:
                    return np.frombuffer(out, dtype=np.int16)
                else:
                    if "503" in err.decode():
                        time.sleep(delay)
                    else:
                        break
            
            return np.frombuffer(out, dtype=np.int16)
        
        if duration is None:
            command = ['ffmpeg', '-y', '-hide_banner', '-ss', str(offset), '-i', str(path), '-f', 's16le', '-ar', str(rate), '-ac', '1', '-'] 
        else:
            command = ['ffmpeg', '-y', '-hide_banner', '-ss', str(offset), '-t', str(duration), '-i', str(path), '-f', 's16le', '-ar', str(rate), '-ac', '1', '-']  

        # Open the audio file with FFmpeg         
        sig = open_audio_with_ffmpeg(command)

        if fmin is not None and fmax is not None:
            sig = audio.bandpass(sig, rate, fmin, fmax)

        return sig, rate
    

if __name__ == "__main__":

    # Create instance of XenoCanto class
    xc = XenoCanto()
    
    # Query Xeno-Canto database and build dataset from metadata
    xc.data(csv_filename='xc_dataset.csv')
    # Label training data and save to cache
    xc.labeling(cache_filename="train_cache.npz")

    # Download audio files from the training dataset
    xc.download()




    # from birdnet_analyzer.train import utils
    # # x_train, y_train, x_test, y_test, valid_labels = utils._load_training_data(cache_mode="load", cache_file="train_data/train_data.npz", progress_callback=None)
    # utils.train_model()