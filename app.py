#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#lINK FOR UPLOADING A FILE FROM WEB PAGE TO PREDICT FNC
# https://www.javatpoint.com/flask-file-uploading

#lINK FOR UPLOADING A FILE FROM WEB PAGE TO PREDICT FNC
# https://stackoverflow.com/questions/58997025/perform-predictions-on-a-data-in-csv-file-using-flask-and-python

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import time

import io
import csv

#FUNCTION TO PREPROCESS DATA : X_TRAIN
def preprocess(train_features_new):
    #ENCODING THE CATEGORICAL DATA
    train_features_new['cp_type'] = train_features_new['cp_type'].map({'trt_cp' : 0 , 'ctl_vehicle' : 1})
    train_features_new['cp_time'] = train_features_new['cp_time'].map({24 : 0 , 72 : 1 , 48 : 2})
    train_features_new['cp_dose'] = train_features_new['cp_dose'].map({'D1' : 0 , 'D2' : 1})
    
    return train_features_new


app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))

# @app.route('/')
# def hello_world():
#     return 'hello_world'


@app.route('/')
def form():
    return """
        <!doctype html>
        <html>
          <head>
            <title>PREDICTION OF MECHANISM OF ACTION</title>
          </head>
          <body>
            <h1>PREDICTION OF MECHANISM OF ACTION</h1>
            <form method="POST" action="/predict" enctype="multipart/form-data">
              <p><input type="file" name="file"></p>
              <p><input type="submit" value="Submit"></p>
            </form>
          </body>
        </html>
    """


@app.route('/index')
def index():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():

    file = request.files['file']
    if not file:
        return "No file"
    data  = pd.read_csv(file , delimiter=',')
    print(data)
    print('THE INPUT IS HAVING TYPE : ' ,type(data))
    print('SUCCESS in READING')
      
    start_time = time.time()
    pipe = pickle.load(open("MODLE_PIPE.pkl", 'rb')) #LOADING MY PRETRAINED MODEL
    data_processed = preprocess(data)
    pred = pipe.predict(data_processed)#INPUT IS A DATAFRAME
    print('FORMAT OF THE PREDICTED OUTPUT : ',type(pred)) #<class 'scipy.sparse.lil.lil_matrix'>
    print('DONE PREDICTION')
    end_time = time.time()
    print('TIME TAKEN TO PREDICT IS : {}'.format(end_time - start_time))
    
    return render_template('index.html',pred='THE GIVEN SHAPE OF INPUT FILE IS {}.\nPREDICTION FOR THE MECHANISMS ARE {}'.format(data.shape  , pred.toarray()))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)

