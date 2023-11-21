import os
import tempfile
from flask import Flask, request, redirect, render_template, url_for
import base64
import numpy as np
import random as rd
import pandas as pd
import joblib
import spacy
from spacy.tokens import Token
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__, template_folder="templates/")
path = "./data/tesis_data.csv"
dfTesis = pd.read_csv(path, delimiter=',')   
path = "./data/normalized_data.csv"
dfNormalized = pd.read_csv(path, delimiter=',')  
dictCarrera = dfTesis.groupby('Facultad')['Carrera'].unique().apply(list).to_dict()
dictTesis = dfTesis.groupby('Carrera')['Titulo'].unique().apply(list).to_dict()
modelo1 = joblib.load("./data/modelo_entrenado_KNNC.pkl")
modelo2 = joblib.load("./data/modelo_entrenado_KNNM.pkl")
modelo3 = joblib.load("./data/modelo_entrenado_NBM.pkl")
modelo4 = joblib.load("./data/modelo_entrenado_LRS.pkl")

def normalize(text1,text2):
    _list = ["the","be"]
    _list2 = ["el","objetivo","proyecto"]
    nlpEs = spacy.load('es_core_news_sm')
    nlpEn = spacy.load('en_core_web_sm')
    for i, res in enumerate(text1):
        res1 = str(res).lower()
        res2 = str(text2[i]).lower()
        if any(word in res.split() for word in _list) and not(any(word in res.split() for word in _list2)):
            doc1 = nlpEn(res1)
            doc2 = nlpEn(res2)
        else:
            doc1 = nlpEs(res1)
            doc2 = nlpEs(res2)
        text1[i] = " ".join([word.lemma_ for word in doc1 if (not word.is_punct)
                                        and (len(word.text) > 2) and (not word.is_stop) and (word.text.strip()!='') and (word.is_alpha)])
        text2[i] = " ".join([word.lemma_ for word in doc2 if (not word.is_punct)
                                        and (len(word.text) > 2) and (not word.is_stop) and (word.text.strip()!='') and (word.is_alpha)])
    return text1, text2

def recommend_function(title,model):
    dfModel = dfNormalized.copy()
    categorical_pipe = Pipeline([
        ('encoder', OneHotEncoder(drop = 'first'))
        ])
    col_transf = ColumnTransformer([
        ('categoric', categorical_pipe, dfModel.select_dtypes('object').columns.tolist())
        ])
    col_transf_fit = col_transf.fit(dfModel)
    text_filtered_transf = col_transf_fit.transform(dfModel)
    text_filtered_transf
    recommendations = list()
    id =  dfTesis.index[dfTesis["Titulo"] == title].tolist()
    if model == "modelo1":
        dif, ind = modelo1.kneighbors(text_filtered_transf[id])
    else:
        dif, ind = modelo2.kneighbors(text_filtered_transf[id])
    for e in ind[0][0:]:
        if e != id:
            recommendations.append(str(dfTesis.loc[e]["Titulo"]))
    return recommendations

def classify_function(title,summary,model):
    summary = [summary]
    title = [title]
    summary, title = normalize(summary,title)
    vectorizer = CountVectorizer(max_features=5200, min_df=3)
    copia = dfNormalized["Resumen normalizado"].values.tolist().copy()
    copia.append(summary[0])
    copia = vectorizer.fit_transform(copia)
    test = copia.toarray()[-1]
    if model == "modelo3":
        y_pred=modelo3.predict(sparse.csr_matrix(test))
    else:
        y_pred=modelo4.predict(sparse.csr_matrix(test))
    return y_pred

@app.route("/")
def main():
    return render_template("index.html")

@app.route("/recommender")
def recommend():
    return render_template("recommender.html", data = dictCarrera)

@app.route("/recommendation", methods=['POST', 'GET'])
def recommendation():
    if request.method == 'POST':
        title = request.form.get('title')
        modelo_seleccionado = request.form.get('modelo')
        result = recommend_function(title,modelo_seleccionado)[1:]
    return render_template("recommendation.html", result = result,title=title)

@app.route("/classifier")
def classify():
    return render_template("classifier.html")

@app.route("/classification", methods=['POST', 'GET'])
def classification():
    if request.method == 'POST':
        title = request.form.get('title')
        summary = request.form.get('summary')
        modelo_seleccionado = request.form.get('modelo')
        result = classify_function(title,summary,modelo_seleccionado)[0]
    return render_template("classification.html", result = result,title=title)

@app.route('/get_careers', methods=['POST'])
def get_careers():
    selected_faculty = request.form['faculty']
    careers = dictCarrera.get(selected_faculty, [])
    options = ''.join(f'<option>{career}</option>' for career in careers)
    options = '<option selected disabled value="">Choose...</option>' + options
    return options

@app.route('/get_tesis', methods=['POST'])
def get_tesis():
    selected_career = request.form['career']
    tesis = dictTesis.get(selected_career, [])
    options = ''.join(f'<option>{career}</option>' for career in tesis)
    options = '<option selected disabled value="">Choose...</option>' + options
    return options