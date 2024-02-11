import sys ,os 
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

# from src.config.configuration import * 

from src.constant import * 
from src.logger import *
from src.exception import * 
from src.config.configuration import * 
from src.components.data_transformation import DataTransformation

from src.components.model_training import ModelTrainer


@dataclass
class DataIngestionConfig():
    raw_file_path = RAW_FILE_PATH
    raw_train_file_path = RAW_TRAIN_FILE_PATH 
    raw_test_file_path = RAW_TEST_FILE_PATH 
    
class DataIngestion():
    def __init__(self):
        self.dataingestionconfig = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        try:
            logging.info('Reading Dataset')
            dataset = pd.read_csv(INITIAL_DATASET_PATH ,encoding="latin-1")
            train_data ,test_data = train_test_split(dataset ,test_size=0.2 ,random_state=121)
            
            
            logging.info('Copy data into raw dataset folder')
            os.makedirs(os.path.dirname(self.dataingestionconfig.raw_file_path) ,exist_ok=True)
            dataset.to_csv(RAW_FILE_PATH ,index=False)
            
            os.makedirs(os.path.dirname(self.dataingestionconfig.raw_train_file_path) ,exist_ok=True)
            train_data.to_csv(RAW_TRAIN_FILE_PATH)
            
            os.makedirs(os.path.dirname(self.dataingestionconfig.raw_test_file_path) ,exist_ok=True)
            test_data.to_csv(RAW_TEST_FILE_PATH ,index =False)
            
            logging.info('Succefully copied data into ingestion folder')
            
            return(
                self.dataingestionconfig.raw_train_file_path ,
                self.dataingestionconfig.raw_test_file_path
            )
        
        except Exception as e:
            logging.info('Unable to ingest data')
            raise CustomException(e ,sys)
        
# if __name__ =='__main__':
#     dataingestion =DataIngestion()
#     train_path ,test_path = dataingestion.initiate_data_ingestion()
#     transform = DataTransformation()
#     transform.initiate_data_transformation(train_path ,test_path)

# if __name__ == '__main__':
#     obj = DataIngestion()
#     train_data ,test_data =obj.initiate_data_ingestion()
#     data_transformer =DataTransformation()
#     train_arr ,test_arr ,_ =data_transformer.initiate_data_transformation(train_data ,test_data)
#     model_training = ModelTrainer()
#     (model_training.initatied_model_traning(train_arr ,test_arr))
