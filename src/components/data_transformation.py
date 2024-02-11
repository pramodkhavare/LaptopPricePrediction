

import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder ,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.sparse import coo_matrix ,hstack ,vstack

from src.constant import *
from src.config.configuration import *
from src.logger import *
from src.exception import *
from src.utils import *


from src.utils import *

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path = PREPROCESSOR_OBJ_FILE_PATH
    transformed_train_path = TRANSFORMED_TRAIN_FILE_PATH
    transformed_test_file_path = TRANSFORMED_TEST_FILE_PATH
    
    


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    
    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initated")
            categorical_column = ['Company', 'TypeName', 'ScreenResolution' ,'OpSys', 'Cpu_brand', 'Gpu_brand']
            numerical_column = ['Inches', 'Ram', 'Weight', 'Touchscreen', 'IPS_panel', 'SSD', 'HDD', 'FlashStorage']
            numericL_pipeline = Pipeline(
                steps=[
                    ('imputer' ,SimpleImputer(strategy='median')),
                    ('scalar' ,StandardScaler())
                    ]
                    )

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer' ,SimpleImputer(strategy='most_frequent')),
                    ('encoder' ,OneHotEncoder(handle_unknown='ignore')),
                    ('scalar' ,StandardScaler(with_mean=False))
                     ]
                    )

            processor = ColumnTransformer(
            [
                    ('num_pipeline' , numericL_pipeline ,numerical_column),
                    ('cat_pipeline' ,categorical_pipeline ,categorical_column)
            ]
                    )
            
            logging.info('Processor created')
            return processor 
        

            
        except Exception as e:
            raise CustomException(e ,sys)
        
    
    def initiate_data_transformation(self ,train_path ,test_path):
        try:
            ## Read Train and Test Data
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read Traning And Test Data Complited")
            logging.info(f'Train Dataframe Head : \n{train_data.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_data.head().to_string()}')
            
            
            
            logging.info("Obtaining Preprosser object")

            preprocessor_obj = self.get_data_transformation_object()


        
    
            target_columns_name = "Price"
            drop_columns = [target_columns_name]
            
            ## spliting dependent and indipend veriable
            input_features_train_data = train_data.drop(drop_columns,axis=1)
            targer_feature_train_data = train_data[[target_columns_name]]
            

            ## spliting dependent and indipend veriable
            input_features_test_data = test_data.drop(drop_columns,axis=1)
            targer_feature_test_data = test_data[[target_columns_name]]
            
            
            ## Apply Transformation object on train and test data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_features_train_data)
            input_feature_test_arr = preprocessor_obj.transform(input_features_test_data)
            

            logging.info("Apply Preprocessor Object on train and test Data")
            
            train_arr = hstack([input_feature_train_arr ,np.array(targer_feature_train_data)]).toarray()
            test_arr = hstack([input_feature_test_arr ,np.array(targer_feature_test_data)]).toarray()
            
            df_train = pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)
            
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_path) ,exist_ok=True)
            df_train.to_csv(self.data_transformation_config.transformed_train_path ,index = False ,header = True) 
            
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_file_path) ,exist_ok=True)
            df_test.to_csv(self.data_transformation_config.transformed_test_file_path ,index = False ,header = True)
            


            ## Callling Save object to save preprocessor pkl file
            save_obj(file_path=self.data_transformation_config.preprocessor_obj_file_path, 
            obj=preprocessor_obj
            )

            logging.info("Preprocessor Object File is Save")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
            
            
        except Exception as e:
            logging.info("Error Occured in initate Data Transformation")
            raise CustomException(e,sys)