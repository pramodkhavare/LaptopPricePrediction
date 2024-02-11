import os ,sys 
import pandas as pd 
import numpy as np 
from datetime import datetime 


def get_time_stamp():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

INITIAL_DATA_DIR = 'notebook\\data'
INITIAL_DATA_DATASET ='clean_laptop.csv'
ROOT_DIRECTORY = os.getcwd()
ARTIFACT_DIR ='Artifacts'

DATA_INGESTION_DIR = 'DataIngestion'
CURRENT_TIME_STAMP =get_time_stamp()
DATA_INGESTION_INGESTED_DATA_DIR = 'IngestedDir'
DATA_INGESTION_INGESTED_DATA_TRAIN_DATASET = 'raw_train.csv'
DATA_INGESTION_INGESTED_DATA_TEST_DATASET = 'raw_test.csv'
DATA_INGESTION_RAW_DATA_DIR = 'Rawdir'
DATA_INGESTION_RAW_DATA_DATASET = 'raw.csv'

DATA_TRANSFORMATION_DIR = 'DataTransformation'
DATA_TRANSFORMATION_PROCESSOR_DIR = 'Processor'
DATA_TRANSFORMATION_PREPROCESSOR_OBJ = 'preprocessor.pkl'
DATA_TRANSFORMATION_FEATURE_ENGINEERING_OBJ = 'feature_eng.pkl'
DATA_TRANSFORMATION_TRANSFORMER_DIR = 'Transfomation'
DATA_TRANSFORMATION_TRANSFORMER_TRAIN_DATASET ='train.csv'
DATA_TRANSFORMATION_TRANSFORMER_TEST_DATASET ='test.csv'

MODEL_TRAINER_DIR = 'Model_Trainer'
MODEL_TRAINER_OBJ = 'model.pkl'








