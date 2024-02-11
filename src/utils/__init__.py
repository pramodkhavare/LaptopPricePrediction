import os ,sys 
import pickle
from sklearn.model_selection import GridSearchCV ,RandomizedSearchCV
from sklearn.metrics import r2_score

from src.exception import *
from src.logger import *


def save_obj(file_path ,obj):
    try:
        os.makedirs(os.path.dirname(file_path) ,exist_ok=True)
        with open (file_path ,'wb') as file_obj:
            pickle.dump(obj ,file_obj)
        
    except Exception as e:
        logging.info('Unable to save object')
        raise CustomException(e ,sys )
    
    
    
    
def load_obj(file_path):
    try:
        with open(file_path  ,'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e ,sys)
    
def evaluate_model(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            #Model Traning
            rs= RandomizedSearchCV(model, para,cv=5)
            rs.fit(X_train,y_train)

            model.set_params(**rs.best_params_)
            model.fit(X_train,y_train)

            #make Prediction
            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test,y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


        

    