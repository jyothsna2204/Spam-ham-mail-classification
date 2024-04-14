from flask import Flask, render_template, request, send_from_directory
import os
app = Flask(__name__)

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
import streamlit as st
load_modell=pickle.load(open('./spam_model2.sav','rb'))
def operation(inp):
    s_input=inp
    #stemmingg=load_modell['function']
    vv=load_modell['v']
    modell=load_modell['model']
    #s_input=stemmingg(s_input)
    s_input=[s_input]
    s_input=vv.transform(s_input)
    oop=modell.predict(s_input)
    if(oop[0]==0):
        return "ham mail"
    else:
        return "spam mail"


@app.route('/')
def index():
    return render_template('spam.html')


@app.route('/predict',methods=['POST'])
def predict():
    inp=request.form['text']
    o=operation(inp)
    return render_template('spam.html', res1=o)

@app.route('/static/<filename>')
def static_files(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)
    
if __name__ == '__main__':
    app.run(debug=True,port=5000)
