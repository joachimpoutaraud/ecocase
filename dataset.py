import os
import subprocess
import numpy as np
import pandas as pd

import json
import tqdm
import time, datetime
import urllib.request
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count

import warnings
warnings.filterwarnings(action="ignore")

def load_audio(path, output_path=None, audio_filter_chain=None, download=False, rate=48000):
    """
    Extracts and filters region of interests (ROIs) from a recording using FFmpeg.

    Parameters:
        path (str): Path to the input audio file.
        audio_filter_chain (str): FFmpeg-compatible audio filter chain string.
        noise_reduction (int): Gain reduction outside the frequency band.
        download (bool): If True, saves filtered audio to disk and returns output path.
        output_folder (str): Folder to save the output audio file.
        encode (str): If 'audio', returns base64-encoded audio. If 'spectrogram', returns base64 image.
        rate (int): Sample rate for encoding and spectrogram.

    Returns:
        np.ndarray or str: Filtered audio data, base64 string, or output path.
    """

    # Get the audio filter chain
    full_filter_chain = ",".join([
        # Resample audio file
        f"aresample=resampler=soxr:out_sample_rate={rate}",
        # Convert to mono
        f"aformat=channel_layouts=mono"
    ])

    if audio_filter_chain is not None:
        full_filter_chain = audio_filter_chain

    if download:
        full_filter_chain += ",asplit=2[a1][a2]"

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-v", "error",
        "-i", path,
        "-filter_complex", full_filter_chain
    ]

    if download:
        cmd += ["-map", "[a1]", "-f", "f32le", "pipe:1", "-map", "[a2]", "-f", "wav", output_path]
    else:
        cmd += ["-f", "f32le", "pipe:1"]

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        audio_bytes, error = process.communicate()
        if error:
            print("FFmpeg error:", error.decode())

    roi_audio = np.frombuffer(audio_bytes, dtype=np.float32)

    if download:
        return roi_audio, output_path

    return [roi_audio]


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
                params_xc = [                 
                    'grp:birds',                 # Group to which the species belongs (birds, bats, frogs, grasshoppers) 
                    'len:"3-60"',                # Length of the recording (seconds)
                    'q:">C"',                    # Quality rating of the recording
                    # 'area:america',              # Continental area: africa, america, asia, australia, europe
                    # 'box:-5.681584,-71.740723,9.535749,-38.364258', # Geographical bounding box (min_lat, min_lon, max_lat, max_lon)
                    # Radius search easily made using bboxfinder.com (coordinate format:"Lat / Lng" + draw a circle with radius ~ 500km, and copy map)
                ],
                species_xc = ['chloris chloris'],
                num_files = None,
                format_time = False,
                format_date = False,
                verbose = True,
                save_path = '',
                overwrite = True,
                multiprocessing = False
                ):

        self.params_xc = params_xc
        self.species_xc = species_xc
        self.num_files = num_files
        self.format_time = format_time
        self.format_date = format_date 
        self.verbose = verbose
        self.save_path = save_path
        self.overwrite = overwrite
        self.multiprocessing = multiprocessing


    def dataset(self, csv_filename=None, download=False):
        """
        Query Xeno-Canto database and build a dataset from the metadata.
        The dataset is a pandas dataframe containing all the recordings metadata matching the Xeno-Canto searching parameters.

        Args:
            csv_filename (str, optional): Name of the CSV file to save the dataset. Default is None.
                If None, the dataset will not be saved.
            download (bool, optional): Whether to download audio files from Xeno-Canto to disk or not. Default is True.

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

        if download:
            self.df = self.download(csv_filename=csv_filename)
        else:
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
            except:
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
                        to_drop = rows.sample(n=len(rows) - self.num_files, random_state=2025).index.to_list()
                        df = df.drop(to_drop)
                    # Test if the number of files is smaller than the minimum number of resquested files for each species 
                    for i in df.en.value_counts()[df.en.value_counts() < self.num_files].index.to_list():
                        to_drop = df[df.en == i].index.to_list()
                        df = df.drop(to_drop)
                        
        # Convert length column to seconds
        def convert_to_seconds(length_str):
            parts = length_str.split(':')
            if len(parts) == 2:
                minutes, seconds = map(int, parts)
                total_seconds = minutes * 60 + seconds
            elif len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
                total_seconds = hours * 3600 + minutes * 60 + seconds
            else:
                raise ValueError(f"Unexpected time format: {length_str}")
            return total_seconds
        
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
    

    def download(self, dataset=None, csv_filename=None):
        """
        Download audio files from XenoCanto.

        Args:
            dataset (str or pd.DataFrame, optional): Path to the dataset CSV file or a pandas DataFrame. Default is None.
            csv_filename (str, optional): The name of the csv file to save the ROIs. Default is None.
            use_multiprocessing (bool, optional): Whether to use multiprocessing for downloading. Default is True.
        """ 
        if self.verbose:
            print('\n====================== DOWNLOAD AUDIO FILES ======================\n')

        # Load dataset if it's a path
        if isinstance(dataset, str):
            ext = os.path.splitext(dataset)[1]
            if ext == ".csv": 
                import pandas as pd
                self.df = pd.read_csv(dataset, sep=";")        

        # Create folders for each species
        if 'categories' in self.df.columns:
            species_list = [s.replace(' ', '_') for s in self.df.categories.unique()]
            for species in species_list:
                os.makedirs(os.path.join(self.save_path, species), exist_ok=True)

        local_file_paths, tasks = [], []

        def prepare_file_path(row):
            if 'start' in row:
                return os.path.join(
                    self.save_path,
                    row['categories'].replace(' ', '_'),
                    f"XC{int(row['id'])}_{float(row['start'])}_{float(row['end'])}.wav"
                )
            else:
                return os.path.join(
                    self.save_path,
                    row['categories'].replace(' ', '_'),
                    f"XC{int(row['id'])}.wav"
                )

        if self.multiprocessing:
            with Pool(processes=max(1, cpu_count() - 1)) as pool:
                for _, row in self.df.iterrows():
                    path = row['file']
                    file_path = prepare_file_path(row)
                    try:
                        tasks.append(pool.apply_async(partial(load_audio, path, output_path=file_path, download=True)))
                        local_file_paths.append(file_path)
                    except:
                        pass

                with tqdm.tqdm(total=len(tasks), desc="Download audio files") as pbar:
                    for task in tasks:
                        task.get()
                        pbar.update(1)
        else:
            with tqdm.tqdm(total=len(self.df), desc="Download audio files") as pbar:
                for _, row in self.df.iterrows():
                    path = row['file']
                    file_path = prepare_file_path(row)
                    try:
                        load_audio(path, output_path=file_path, download=True)
                        local_file_paths.append(file_path)
                    except:
                        pass
                    pbar.update(1)

        # Save updated dataframe if requested
        if csv_filename is not None:
            from pathlib import Path
            save_path = Path(self.save_path)
            csv_filename = save_path / csv_filename
            self.df['file'] = local_file_paths         
            self.df.to_csv(csv_filename, index=False, sep=";")

        return self.df
    

if __name__ == "__main__":

    xc = XenoCanto()
    xc.dataset(download=True)
