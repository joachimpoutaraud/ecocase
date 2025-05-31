import os
import pandas as pd
import numpy as np

import json
import tqdm
import time, datetime
import urllib.request
from pathlib import Path
from multiprocessing.pool import Pool

from birdnet_analyzer import audio
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


    def open(self, path: str, rate=cfg.SAMPLE_RATE, offset=0.0, duration=None, fmin=None, fmax=None):
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
                    return np.frombuffer(out, dtype='float32')
                else:
                    if "503" in err.decode():
                        time.sleep(delay)
                    else:
                        break
            
            return np.frombuffer(out, dtype='float32')
        
        if duration is None:
            command = [
                'ffmpeg', '-y', '-hide_banner', '-ss', str(offset), '-i', str(path),
                '-f', 'f32le', '-ar', str(rate), '-ac', '1', '-'
            ]
        else:
            command = [
                'ffmpeg', '-y', '-hide_banner', '-ss', str(offset), '-t', str(duration), '-i', str(path),
                '-f', 'f32le', '-ar', str(rate), '-ac', '1', '-'
            ]

        # Open the audio file with FFmpeg         
        sig = open_audio_with_ffmpeg(command)

        if fmin is not None and fmax is not None:
            sig = audio.bandpass(sig, rate, fmin, fmax)

        return sig, rate
    
    
    def write(self, row):
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
                sig, rate = self.open(row['file'], cfg.SAMPLE_RATE, row.start, duration=row.end - row.start)
                output_file = os.path.join(os.path.join(self.save_path, row['categories'].replace(' ', '_')), f"XC{int(id_number)}_{float(row['start'])}_{float(row['end'])}.wav")
            else:
                sig, rate = self.open(row['file'], cfg.SAMPLE_RATE)
                output_file = os.path.join(os.path.join(self.save_path, row['categories'].replace(' ', '_')), f"XC{int(id_number)}.wav")

            sf.write(output_file, sig, rate, format='wav')
            
            # Return updated file path
            return output_file
            
        except Exception as ex:
            print(f"Error: Cannot analyze audio file {row.file}.", flush=True)
            from birdnet_analyzer import utils
            utils.write_error_log(ex)
    

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
                for output_file in pool.imap(self.write, rows):
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