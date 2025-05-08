import os

#################
#  CONFIG FILE  #
#################

RANDOM_SEED: int = 2025         # Fixed random seed

#########################
# XENO-CANTO PARAMETERS #
#########################

# Number of audio files to download for each species
NUM_FILES: int = 100          
# Xeno-Canto scientific species (e.g. ['Columba palumbus','Fringilla coelebs']) 
# If None, species will be requested according to the Xeno-Canto searching parameters
SPECIES_XC: list[str] = [
    # TINAMOUS
    'Tinamus major',
    'Crypturellus brevirostris',
    'Crypturellus soui',
    'Crypturellus cinereus',
    'Crypturellus variegatus',
    # POTOOS
    'Nyctibius grandis',
    'Nyctibius griseus',
    'Nyctibius bracteatus',
    'Nyctibius aethereus',
    'Nyctibius leucopterus', 
    # FALCONS
    'Micrastur ruficollis',
    'Micrastur gilvicollis',
    'Micrastur mirandollei',
    'Micrastur semitorquatus', 
    # OWLS
    'Megascops roraimae',
    'Megascops watsonii', 
    'Lophostrix cristata',
    'Pulsatrix perspicillata',
    'Strix virgata',
    'Strix huhula',
    'Glaucidium hardyi',
    # NIGHTJARS
    'Lurocalis semitorquatus',
    'Hydropsalis maculicaudus',
    'Hydropsalis cayennensis',
    'Nyctiprogne leucopyga',
    'Chordeiles acutipennis',
    'Chordeiles nacunda',
    'Nyctipolus nigrescens', 
    'Nyctidromus albicollis',
    'Hydropsalis climacocerca'
]     
# Xeno-Canto searching parameters (https://www.xeno-canto.org/help/search)
PARAMS_XC: list[str] = [                 
    'grp:birds',                # Group to which the species belongs (birds, bats, frogs, grasshoppers) 
    # 'type:song',                # Type of the bird sound
    'len:"<100"',               # Length of the recording (seconds)
    'q:">D"',                   # Quality rating of the recording
    'area:america',             # Continental area: africa, america, asia, australia, europe
    # 'box:-5.681584,-71.740723,9.535749,-38.364258', # Geographical bounding box (min_lat, min_lon, max_lat, max_lon)
    # Radius search easily made using bboxfinder.com (coordinate format:"Lat / Lng" + draw a circle with radius ~ 500km, and copy map)
] 

##########################
# MODEL PATHS AND CONFIG #
##########################

MODEL_VERSION: str = "V2.4"
PB_MODEL: str = "checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model"
MODEL_PATH: str = "checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite" 
LABELS_FILE: str = "checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels.txt"
MDATA_MODEL_PATH: str = "checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite"
TRANSLATED_LABELS_PATH: str = "labels/V2.4"

# Path to custom trained classifier
# If None, no custom classifier will be used
# Make sure to set the LABELS_FILE above accordingly
CUSTOM_CLASSIFIER = None

##################
# AUDIO SETTINGS #
##################

# We use a sample rate of 48kHz, so the model input size is
# (batch size, 48000 kHz * 3 seconds) = (1, 144000)
# Recordings will be resampled automatically.
SAMPLE_RATE: int = 48000
# We're using 3-second chunks
SIG_LENGTH: float = 3.0
# Define overlap between consecutive chunks < 3.0
SIG_OVERLAP: float = 0.0
# Define minimum length of audio chunk for prediction,
# chunks shorter than 3 seconds will be padded with zeros
SIG_MINLEN: float = 1.0
# Frequency range. This is model specific and should not be changed.
SIG_FMIN: int = 0
SIG_FMAX: int = 15000
# Settings for bandpass filter
BANDPASS_FMIN: int = 150
BANDPASS_FMAX: int = 5000

#########################
#  CLUSTERING SETTINGS  #
#########################

MIN_PTS = 'auto'                  # number of samples in a neighborhood for a point to be considered as a core point {int or 'auto'}
EPS = 'auto'                      # maximum distance between two samples for one to be considered as in the neighborhood of the other {float or 'auto'}
METRIC: str = 'cosine'            # metric to use when calculating distance between instances in a feature array.
METHOD: str = 'HDBSCAN'           # cluster method that allow some points beeing outside of a cluster {HDBSCAN, DBSCAN}
KEEP: str = 'ALL'                 # define if we want to keep all the clusters or only the biggest cluster, while the rest is discarded as noise {BIGGEST, ALL}

#####################
# METADATA SETTINGS #
#####################

LATITUDE: float = -1.0
LONGITUDE: float = -1.0
WEEK: int = -1
LOCATION_FILTER_THRESHOLD: float = 0.03

######################
# INFERENCE SETTINGS #
######################

# If None or empty file, no custom species list will be used
# Note: Entries in this list have to match entries from the LABELS_FILE
# We use the 2021 eBird taxonomy for species names (Clements list)
CODES_FILE: str = "checkpoints/eBird_taxonomy_codes_2021E.json"
SPECIES_LIST_FILE: str = "example/species_list.txt"

# File input path and output path for selection tables
INPUT_PATH: str = "example/"
OUTPUT_PATH: str = "example/"

# Supported file types
ALLOWED_FILETYPES: list[str] = ["wav", "flac", "mp3", "ogg", "m4a", "wma", "aiff", "aif"]

# Number of threads to use for inference.
# Can be as high as number of CPUs in your system
CPU_THREADS: int = os.cpu_count()
TFLITE_THREADS: int = 1

# False will output logits, True will convert to sigmoid activations
APPLY_SIGMOID: bool = True
SIGMOID_SENSITIVITY: float = 1.0

# Minimum confidence score to include in selection table
# (be aware: if APPLY_SIGMOID = False, this no longer represents probabilities and needs to be adjusted)
MIN_CONFIDENCE: float = 0.7

# Number of samples to process at the same time. Higher values can increase
# processing speed, but will also increase memory usage.
# Might only be useful for GPU inference.
BATCH_SIZE: int = 1

# Number of seconds to load from a file at a time
# Files will be loaded into memory in segments that are only as long as this value
# Lowering this value results in lower memory usage
FILE_SPLITTING_DURATION: int = 600

# Whether to use noise to pad the signal
# If set to False, the signal will be padded with zeros
USE_NOISE: bool = False

# Specifies the output format. 'table' denotes a Raven selection table,
# 'audacity' denotes a TXT file with the same format as Audacity timeline labels
# 'csv' denotes a generic CSV file with start, end, species and confidence.
RESULT_TYPES: set[str] | list[str] = {"table"}
OUTPUT_RAVEN_FILENAME: str = "BirdNET_SelectionTable.txt"  # this is for combined Raven selection tables only
OUTPUT_RTABLE_FILENAME: str = "BirdNET_RTable.csv"
OUTPUT_KALEIDOSCOPE_FILENAME: str = "BirdNET_Kaleidoscope.csv"
OUTPUT_CSV_FILENAME: str = "BirdNET_CombinedTable.csv"

# Whether to skip existing results in the output path
# If set to False, existing files will not be overwritten
SKIP_EXISTING_RESULTS: bool = False
COMBINE_RESULTS: bool = False

#####################
# TRAINING SETTINGS #
#####################

# Training data path
TRAIN_DATA_PATH = None
TEST_DATA_PATH = None

# List of non-event classes
NON_EVENT_CLASSES: list[str] = ["noise", "other", "background", "silence"]

# Sample crop mode
SAMPLE_CROP_MODE: str = "center"

# Upsampling settings
UPSAMPLING_RATIO: float = 0.0
UPSAMPLING_MODE = "repeat"

# Number of epochs to train model
TRAIN_EPOCHS: int = 150            

# Batch size for training model
TRAIN_BATCH_SIZE: int = 16

# Learning rate for training model
TRAIN_LEARNING_RATE: float = 2e-4

# Number of hidden units in custom classifier
# If >0, a two-layer classifier will be trained
TRAIN_HIDDEN_UNITS: int = 512

# Dropout rate for training
TRAIN_DROPOUT: float = 0.75

# Validation split (percentage)
TRAIN_VAL_SPLIT: float = 0.2

# Whether to use mixup for training
TRAIN_WITH_MIXUP: bool = True

# Whether to apply label smoothing for training
TRAIN_WITH_LABEL_SMOOTHING: bool = True

# Model output format
TRAINED_MODEL_OUTPUT_FORMAT: str = "tflite"

# Model save mode (replace or append new classifier)
TRAINED_MODEL_SAVE_MODE: str = "replace"

# Cache settings
TRAIN_CACHE_MODE: str = "none"
TRAIN_CACHE_FILE: str = "train_cache.npz"

# Use automatic Hyperparameter tuning
AUTOTUNE: bool = True

# How many trials are done for the hyperparameter tuning
AUTOTUNE_TRIALS: int = 10

# How many executions per trial are done for the hyperparameter tuning
# Mutliple executions will be averaged, so the evaluation is more consistent
AUTOTUNE_EXECUTIONS_PER_TRIAL: int = 1

# If a binary classification model is trained.
# This value will be detected automatically in the training script, if only one class and a non-event class is used.
BINARY_CLASSIFICATION: bool = False

# If a model for a multi-label setting is trained.
# This value will automatically be set, if subfolders in the input directory are named with multiple classes separated by commas.
MULTI_LABEL: bool = True

#####################
# MISC RUNTIME VARS #
#####################

CODES = {}
LABELS: list[str] = []
TRANSLATED_LABELS: list[str] = []
SPECIES_LIST: list[str] = []
ERROR_LOG_FILE: str = "error_log.txt"
FILE_LIST = []
FILE_STORAGE_PATH: str = ""

######################
# GET AND SET CONFIG #
######################

def get_config():
    return {k: v for k, v in globals().items() if k.isupper()}

def set_config(c: dict):
    for k, v in c.items():
        globals()[k] = v