#importing required libraries
import pandas
from flask import Flask, request, render_template,url_for,redirect
import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
from fileinput import filename
warnings.filterwarnings('ignore')
from feature import FeatureExtraction
import os
from werkzeug.utils import secure_filename



file = open("pickle/model.pkl","rb")
gbc = pickle.load(file)
file.close()


app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def login():
    
    if request.method == "POST":
        first_name = request.form.get("username")
        last_name = request.form.get("password") 
        if first_name == 'admin' and last_name == 'admin':
            return redirect("/home")
    else:
        return render_template("login.html")

@app.route("/home", methods=["GET", "POST"])
def home():
    return render_template("home.html")
@app.route("/detect/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30) 

        y_pred =gbc.predict(x)[0]
        #1 is safe       
        #-1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0,0]
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
        return render_template('index.html',xx =round(y_pro_non_phishing,2),url=url )
    return render_template("index.html", xx =-1)

if __name__ == "__main__":
    app.run(debug=True)