import json
import time, datetime
import urllib.request
from pathlib import Path
import pandas as pd

import config as cfg

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
                save_path = None,
                ):

        self.params_xc = params_xc
        self.species_xc = species_xc
        self.num_files = num_files
        self.format_time = format_time
        self.format_date = format_date 
        self.verbose = verbose
        self.save_path = save_path
        

    def dataset(self, data_path=None, csv_filename=None):

        """
        Query Xeno-Canto database and build a dataset from the metadata.
        The dataset is a pandas dataframe containing all the recordings metadata matching the Xeno-Canto searching parameters.

        Args:
            data_path (str, optional): Path to the directory where the data is saved. Default is None.
            csv_filename (str): Name of the csv file to save the dataset. Default is None.
                If None, the csv file is not saved. 

        Returns:
            pandas.DataFrame: Dataframe containing all the recordings metadata matching the Xeno-Canto searching parameters.
        """

        if data_path is not None:
            #TODO Load audio data with categories from a folder to a dataframe
            # https://github.com/birdnet-team/BirdNET-Analyzer/blob/2c68450fbee102cea1a1ed08a503d779c15c41bc/birdnet_analyzer/train/utils.py#L105
            pass
        
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
                print('   - Number of species: %2.0f' % len(self.df['en'].unique()))
                print('   - Unique species: {}'.format(self.df['en'].unique()))
            else:
                print('\n*** WARNING ***\nThe dataframe is empty.')   
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
    

    def _load_training_data(cache_mode=None, cache_file="", progress_callback=None):
        """Loads the data for training.

        Reads all subdirectories of "config.TRAIN_DATA_PATH" and uses their names as new labels.

        These directories should contain all the training data for each label.

        If a cache file is provided, the training data is loaded from there.

        Args:
            cache_mode: Cache mode. Can be 'load' or 'save'. Defaults to None.
            cache_file: Path to cache file.

        Returns:
            A tuple of (x_train, y_train, x_test, y_test, labels).
        """

        import os
        import numpy as np
        import utils

        # Load from cache
        if cache_mode == "load":
            if os.path.isfile(cache_file):
                print(f"\t...loading from cache: {cache_file}", flush=True)
                x_train, y_train, x_test, y_test, labels, cfg.BINARY_CLASSIFICATION, cfg.MULTI_LABEL = (
                    utils.load_from_cache(cache_file)
                )
                return x_train, y_train, x_test, y_test, labels
            else:
                print(f"\t...cache file not found: {cache_file}", flush=True)

        # Print train and test data path as confirmation
        print(f"\t...train data path: {cfg.TRAIN_DATA_PATH}", flush=True)
        print(f"\t...test data path: {cfg.TEST_DATA_PATH}", flush=True)

        # Get list of subfolders as labels
        train_folders = list(sorted(utils.list_subdirectories(cfg.TRAIN_DATA_PATH)))

        # Read all individual labels from the folder names
        labels = []

        for folder in train_folders:
            labels_in_folder = folder.split(",")
            for label in labels_in_folder:
                if label not in labels:
                    labels.append(label)

        # Sort labels
        labels = list(sorted(labels))

        # Get valid labels
        valid_labels = [
            label for label in labels if label.lower() not in cfg.NON_EVENT_CLASSES and not label.startswith("-")
        ]

        # Check if binary classification
        cfg.BINARY_CLASSIFICATION = len(valid_labels) == 1

        # Validate the classes for binary classification
        if cfg.BINARY_CLASSIFICATION:
            if len([f for f in train_folders if f.startswith("-")]) > 0:
                raise Exception(
                    "Negative labels can't be used with binary classification",
                    "validation-no-negative-samples-in-binary-classification",
                )
            if len([f for f in train_folders if f.lower() in cfg.NON_EVENT_CLASSES]) == 0:
                raise Exception(
                    "Non-event samples are required for binary classification",
                    "validation-non-event-samples-required-in-binary-classification",
                )

        # Check if multi label
        cfg.MULTI_LABEL = len(valid_labels) > 1 and any("," in f for f in train_folders)

        # Check if multi-label and binary classficication
        if cfg.BINARY_CLASSIFICATION and cfg.MULTI_LABEL:
            raise Exception("Error: Binary classfication and multi-label not possible at the same time")

        # Only allow repeat upsampling for multi-label setting
        if cfg.MULTI_LABEL and cfg.UPSAMPLING_RATIO > 0 and cfg.UPSAMPLING_MODE != "repeat":
            raise Exception("Only repeat-upsampling ist available for multi-label", "validation-only-repeat-upsampling-for-multi-label")

        # Load training data
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        def load_data(data_path, allowed_folders):
            x = []
            y = []
            
            folders = list(sorted(utils.list_subdirectories(data_path)))

            for folder in folders:
                if folder not in allowed_folders:
                    print(f"Skipping folder {folder} because it is not in the training data.", flush=True)
                    continue

                # Get label vector
                label_vector = np.zeros((len(valid_labels),), dtype="float32")
                folder_labels = folder.split(",")

                for label in folder_labels:
                    if label.lower() not in cfg.NON_EVENT_CLASSES and not label.startswith("-"):
                        label_vector[valid_labels.index(label)] = 1
                    elif (
                        label.startswith("-") and label[1:] in valid_labels
                    ):  # Negative labels need to be contained in the valid labels
                        label_vector[valid_labels.index(label[1:])] = -1

                # Get list of files
                # Filter files that start with '.' because macOS seems to them for temp files.
                files = filter(
                    os.path.isfile,
                    (
                        os.path.join(data_path, folder, f)
                        for f in sorted(os.listdir(os.path.join(data_path, folder)))
                        if not f.startswith(".") and f.rsplit(".", 1)[-1].lower() in cfg.ALLOWED_FILETYPES
                    ),
                )

                # Load files using thread pool
                with Pool(cfg.CPU_THREADS) as p:
                    tasks = []

                    for f in files:
                        task = p.apply_async(
                            partial(_load_audio_file, f=f, label_vector=label_vector, config=cfg.get_config())
                        )
                        tasks.append(task)

                    # Wait for tasks to complete and monitor progress with tqdm
                    num_files_processed = 0

                    with tqdm.tqdm(total=len(tasks), desc=f" - loading '{folder}'", unit="f") as progress_bar:
                        for task in tasks:
                            result = task.get()
                            # Make sure result is not empty
                            # Empty results might be caused by errors when loading the audio file
                            # TODO: We should check for embeddings size in result, otherwise we can't add them to the training data
                            if len(result[0]) > 0:
                                x += result[0]
                                y += result[1]

                            num_files_processed += 1
                            progress_bar.update(1)

                            if progress_callback:
                                progress_callback(num_files_processed, len(tasks), folder)

            return np.array(x, dtype="float32"), np.array(y, dtype="float32")

        x_train, y_train = load_data(cfg.TRAIN_DATA_PATH, train_folders)

        if cfg.TEST_DATA_PATH and cfg.TEST_DATA_PATH != cfg.TRAIN_DATA_PATH:
            test_folders = list(sorted(utils.list_subdirectories(cfg.TEST_DATA_PATH)))
            allowed_test_folders = [
                folder for folder in test_folders if folder in train_folders and not folder.startswith("-")
            ]
            x_test, y_test = load_data(cfg.TEST_DATA_PATH, allowed_test_folders)
        else:
            x_test = np.array([])
            y_test = np.array([])

        # Save to cache?
        if cache_mode == "save":
            print(f"\t...saving training data to cache: {cache_file}", flush=True)
            try:
                # Only save the valid labels
                utils.save_to_cache(cache_file, x_train, y_train, x_test, y_test, valid_labels)
            except Exception as e:
                print(f"\t...error saving cache: {e}", flush=True)

        # Return only the valid labels for further use
        return x_train, y_train, x_test, y_test, valid_labels