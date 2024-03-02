from src.constant import *
from src.config.configuration import * 
from src.utils import *
from src.logger import logging 
from src.exception import CustomException 




    
class CustomData():
    def __init__(self ,
                 Company :str,
                 TypeName :str,
                 OpSys:str,
                 Cpu_brand:str,
                 Gpu_brand:str,
                 Ram :int,
                 Weight :int,
                 Touchscreen ,
                 IPS_panel ,
                 SSD ,
                 HDD ,
                 FlashStorage ,
                 ScreenResolution,
                 Inches
                 
                 ):
        self.Company =Company 
        self.TypeName =TypeName
        self.OpSys =OpSys 
        self.Cpu_brand =Cpu_brand
        self.Gpu_brand =Gpu_brand
        self.Ram =Ram
        self.Weight = Weight 
        self.Touchscreen =Touchscreen 
        self.IPS_panel =IPS_panel
        self.SSD =SSD
        self.HDD = HDD
        self.FlashStorage =FlashStorage
        self.ScreenResolution =ScreenResolution
        self.Inches =Inches
        
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Company' : [self.Company],
                'TypeName' :[self.TypeName],
                'OpSys' :[self.OpSys],
                'Cpu_brand' :[self.Cpu_brand],
                'Gpu_brand' : [self.Gpu_brand],
                'Ram' : [self.Ram],
                'Weight' : [self.Weight],
                'Touchscreen' : [self.Touchscreen],
                'IPS_panel' : [self.IPS_panel],
                'SSD' :[self.SSD],
                'HDD' : [self.HDD],
                'FlashStorage' : [self.FlashStorage],
                'Inches' : [self.Inches] ,
                'ScreenResolution' : [self.ScreenResolution]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Data Gathered')
            return df
        except Exception as e:
            raise CustomException(e ,sys)
    
    
    
class Prediction():
    def __init__(self):
        pass 
    
    def prediction(self ,features):
        try:
            logging.info('Prediction started')
            preprocessor_file_path  = PREPROCESSOR_OBJ_FILE_PATH
            model_file_path = MODEL_TRAINER_FILE_PATH 
            preprocessor = load_obj(preprocessor_file_path)
            model = load_obj(model_file_path)
            
            
            scaled_data = preprocessor.transform(features)
            pred = model.predict(scaled_data)
        
            logging.info('Succefully predicted output')
            
            return pred 
        except Exception as e:
            logging.info('Unable to predict output')
            raise(CustomException(e ,sys))