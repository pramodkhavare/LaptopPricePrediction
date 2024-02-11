import os ,sys 
from dataclasses import dataclass 
import os ,sys 
from dataclasses import dataclass  
import numpy as np 
import pandas as pd 
import math
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


from src.constant import * 
from src.config.configuration import *
from src.utils import * 
from src.logger import logging 
from src.exception import CustomException 


class ModelTrainingConfig():
    model_trainer_file = MODEL_TRAINER_FILE_PATH
    
class ModelTrainer():
    def __init__(self):
        self.modeltrainingconfig = ModelTrainingConfig()
        
    def initatied_model_traning(self,train_array,test_array):
        try:
            logging.info("Split Dependent And Independent Features")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "LinearRegression":LinearRegression(),
                "Ridge":Ridge(),
                "Lesso":Lasso(),
                "Elastic":ElasticNet(),
                "SVR":SVR(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "ExtraTreeRegressor":ExtraTreeRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "BaggingRegressor":BaggingRegressor(GradientBoostingRegressor())
            }

            params = {
                "LinearRegression":{
                    
                },
                "Lesso":{
                    "alpha": [0.01, 0.1, 1, 10,20]
                },
                 "Ridge":{
                    "alpha": [0.01, 0.1, 1, 10,20]
                },
                "Elastic":{
                    "alpha": [0.01, 0.1, 1, 10],
                    "l1_ratio": [0.2, 0.4, 0.6, 0.8]
                },
                "DecisionTreeRegressor":{
                    "criterion":["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    "splitter":['best','random'],
                    "max_depth": [3, 5, 7, 9, 11],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features":["auto","sqrt","log2"]
                },
                "ExtraTreeRegressor":{
                    "criterion":["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    "splitter":['best','random'],
                    "max_depth": [8,12,13,20,25],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features":["auto","sqrt","log2"],
                },
                "RandomForestRegressor":{
                    "criterion":["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    'n_estimators': [100,150,300],
                    'max_depth': [8,10,15,20],
                    'min_samples_split': [0.5,4, 5],
                    'min_samples_leaf': [3, 5, 6],
                },
                "AdaBoostRegressor":{
                    'n_estimators': [ 180, 200,300],
                    "learning_rate":[0.1,0.001,0.01,1,0.00001],
                    "loss":["linear", "square", "exponential"]
                },
                "SVR":{
                    "gamma":["scale", "auto"],
                    "C": [0.01, 0.1, 1, 10],
                },
                "GradientBoostingRegressor":{
                    'n_estimators': [ 180, 200,300,400],
                    "learning_rate":[0.1,0.001,0.01,1,0.00001],
                    "loss":["squared_error", "absolute_error", "huber", "quantile"],
                    "max_depth": [8,10,15,20,30],
                    "min_samples_split": [8,10,6,20,25],
                    "min_samples_leaf": [5,6,8,10,15,20],
                    "max_features":["auto","sqrt","log2"],
                },
                "KNeighborsRegressor":{
                    "n_neighbors":[8,10,15,18],
                    "algorithm":["auto", "ball_tree", "kd_tree", "brute"],
                    "leaf_size":[35,40,45,50],
                },
                "BaggingRegressor":{

                },
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                models=models,param=params)

                ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))


            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")
            print("\n***************************************************************************************\n")
            logging.info(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")

            save_obj(file_path=self.modeltrainingconfig.model_trainer_file,
                obj = best_model
                )

        except Exception as e:
            logging.info("Error Occured in Model Traning")
            raise CustomException(e,sys)