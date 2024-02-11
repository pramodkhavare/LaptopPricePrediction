from src.constant import * 
from src.config.configuration import *
from src.logger import logging 
from src.exception import CustomException 
from src.pipeline.prediction_pipeline import Prediction ,CustomData
from src.pipeline.training_pipeline import Train 

import os ,sys 
from flask import Flask ,render_template,request
import sys 
import numpy as np 

application = Flask(__name__)
app = application

@app.route("/predict",methods = ["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")

    else:
        data = CustomData(
             Company = request.form.get("Company")
            ,TypeName = request.form.get("TypeName")
            ,Ram = int(request.form.get("Ram"))
            ,OpSys = request.form.get("OpSys")
            ,Weight = float(request.form.get("Weight"))
            ,Touchscreen = int(request.form.get("Touchscreen"))
            ,ScreenResolution =request.form.get('ScreenResolution')
            ,FlashStorage =request.form.get("FlashStorage")
            ,IPS_panel = int(request.form.get("IPS_panel"))
            ,Cpu_brand = request.form.get("Cpu_brand")
            ,HDD = int(request.form.get("HDD"))
            ,SSD = int(request.form.get("SSD"))
            ,Gpu_brand =  request.form.get("Cpu_brand")
            ,Inches = request.form.get("Inches")
            
        )



        final_data = data.get_data_as_dataframe()
        predict_pipline = Prediction()
        pred = predict_pipline.prediction(final_data)
        pred = pred 
        result = np.round(pred ,2)[0] 
        
        return render_template("form.html",final_result = "Your Laptop  Price Is: {}".format(result))
        
        
        
        
if __name__ == '__main__':
    app.run(host='0.0.0.0' ,debug=True , port='8000')