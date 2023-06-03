import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import Parallel, delayed
import joblib

app=Flask(__name__)
tfvect= joblib.load(open("E:/Techbytes/Fake news detection Final Project/Models/vectoriser.pkl","rb"))
model = joblib.load(open("E:/Techbytes/Fake news detection Final Project/Models/RFModel.pkl","rb"))
modelLR= joblib.load(open("E:/Techbytes/Fake news detection Final Project/Models/LRModel.pkl","rb"))
modelDT= joblib.load(open("E:/Techbytes/Fake news detection Final Project/Models/DTModel.pkl","rb"))
modelGB= joblib.load(open("E:/Techbytes/Fake news detection Final Project/Models/GBModel.pkl","rb"))
from sklearn.feature_extraction.text import TfidfVectorizer

def fake_news_detRF(vectorized_input_data):
    prediction = model.predict(vectorized_input_data)
    return prediction

def fake_news_detLR(vectorized_input_data):
    prediction = modelLR.predict(vectorized_input_data)
    return prediction

def fake_news_detDT(vectorized_input_data):
    prediction = modelDT.predict(vectorized_input_data)
    return prediction



@app.route("/")
def Home():
    return render_template("home.html")
    
@app.route("/submit", methods=['GET','POST'])
def predict():
    if request.method=='POST':
        news=request.form['title']
        if len(news)<50:
            return render_template("Home.html",prediction=2)
        input_data = [news]
        vectorized_input_data = tfvect.transform(input_data)
        pred = fake_news_detRF(vectorized_input_data)
        pred1 = fake_news_detDT(vectorized_input_data)
        pred2 = fake_news_detLR(vectorized_input_data)
        print(pred)
        if pred==1 or pred1==1 or pred2==1:
            return render_template("Home.html",prediction=1)
        else:
            return render_template("Home.html",prediction=0)
    return render_template("Home.html",prediction="")

if __name__=='__main__':
    app.run(debug=True)

