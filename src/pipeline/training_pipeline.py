from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer 
import os,sys 
import pandas as pd 
import numpy as np 


class Train():
    def __init__(self):
        pass 
    
    def main(self):
        obj = DataIngestion()
        train_data ,test_data =obj.initiate_data_ingestion()
        data_transformer =DataTransformation()
        train_arr ,test_arr ,_ =data_transformer.initiate_data_transformation(train_data ,test_data)
        model_training = ModelTrainer()
        (model_training.initatied_model_traning(train_arr ,test_arr))

if __name__ == '__main__':
    obj = Train()
    obj.main()

# # Run
# if __name__=="__main__":
#     obj = DataIngestion()
#     train_data_path,test_data_path = obj.initated_data_ingestion()
#     data_transformation = DataTransformation()
#     train_arr,test_arr,_ = data_transformation.initatie_data_transformation(train_data_path, test_data_path)
#     model_traning = ModelTraning()
#     model_traning.initatied_model_traning(train_arr, test_arr)
