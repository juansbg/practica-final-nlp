# Importamos las librerias mínimas necesarias
from os import name, path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dash_table
#import dash_core_components as dcc
#import dash_html_components as html
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import logging
from plotly.subplots import make_subplots
import plotly.express as px
import base64
import plotly.io as pio
from joblib import load

import matplotlib.pyplot as plt
# import seaborn as sns
import re
# import tensorflow as tf
from joblib import load

app = dash.Dash()
server = app.server

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1)[:, None])
    return e_x / np.sum(e_x, axis=1)[:, None]

def sentiment_analysis(text_payload):
    try:
        inputs = tokenizer(text_payload, return_tensors="pt")["input_ids"]
        logits = model(inputs).logits
        return softmax(np.array(logits.detach()))[0]
    except:
        return [1/3,1/3,1/3]

app.layout = html.Div([
    html.Div(
        [
            html.H1( # Primera fila
                children = [
                    'Sentiment Analysis of Bank Statement'
                    #"""
                ],
                id = "titulo",
                style = {  # Aquí aplico todo lo que necesite de CSS
                    "text-align": "center", # Alineo el texto al centro
                    "font-size": "50px",
                    "font-weight": "bold",
                    "-webkit-text-fill-color": "transparent",
                    "text-fill-color": "transparent",
                    "-webkit-background-clip": "text",
                    "background-clip": "text",
                    "background-image": "linear-gradient(90deg, #2874a6 , #633974 , #b03a2e )"
                }
            )
        ],
        style={
            "width":'1000px',
            "margin":"auto",
        }
    ),
    html.Div(
        dcc.Tabs(id="tabs-styled-with-props", value='tab-1', children=[
            dcc.Tab(label='Sentence Sentiment', value='tab-1'),
            dcc.Tab(label='PDF Report', value='tab-2'),
            #dcc.Tab(label='Predicción', value='tab-3')
        ], colors={
            "primary": "#58d68d",
            #"background": "#f7dc6f",
        }),
        style={
            "font-weight": "bold",
            "font-size": "30px",
            "background-image": "linear-gradient(90deg,#58d68d,#f4d03f,#ec7063)",
            "color": "#323232",
            "margin-bottom":"40px",
        }
    ),
    html.Div(id='content-div',children=[], style={"width": '80%',
              "margin": "auto",
              "display":"block"
              })
], 
    style={
        "font-family": '"Century Gothic", CenturyGothic, Geneva, AppleGothic, sans-serif',
        # "background-color": "#323232",
        "color": "#323232",
    }
)

@app.callback(
    Output('content-div', 'children'),
    Input('tabs-styled-with-props', 'value')
)
def change_tab(tab):
    if tab == 'tab-1':
        content = [
            dcc.Textarea(
                id='textarea-example',
                placeholder='Financial sentence...',
                style={'width': '50%', "font-size": "20px", 'height': 100},
            ),
            html.Div(id='textarea-example-output',
                    style={'whiteSpace': 'pre-line', 'width': '48%', "margin-left": "2%", 'height': 300, })
        ]
    if tab == 'tab-2':
        content = [
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop PDF or ',
                    html.A('click to Select PDF', style={'color':'#3498db','font-weight':'bold', 'cursor':'pointer', 'text-decration':'underline'})
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),
            html.Div(
                    id='output-data-upload', 
                    children=[
                        html.Div(id='output-section-1', children="", style={'width':'33%', 'backgound-color':'red'}),
                        html.Div(id='output-section-2', children="", style={'width':'33%', 'margin-right':'30px','backgound-color':'blue'}),
                        html.Div(id='output-section-3', children="", style={'width':'33%', 'backgound-color':'green'}),
                    ],
                    style={
                        'width':'100%', 'display':'flex',
                    }
                ),
        ]
    return content

@app.callback(
    Output('textarea-example-output', 'children'),
    Input('textarea-example', 'value')
)
def update_output(value):
    neg, neut, pos = sentiment_analysis(value)
    result = html.Div(children=[
        html.Div(children=[html.H3('Positive: ', style={"-webkit-text-fill-color": "transparent",
                    "text-fill-color": "transparent",
                    "-webkit-background-clip": "text",
                    "background-clip": "text",
                    "background-image": "linear-gradient(90deg,  #28b463,  #48c9b0)"}), 
                    html.H3(f'{round(float(pos),4)} %'), ], style={'width': '33%', 'float': 'left'}),
        html.Div(children=[html.H3('Neutral: ', style={"-webkit-text-fill-color": "transparent",
                    "text-fill-color": "transparent",
                    "-webkit-background-clip": "text",
                    "background-clip": "text",
                    "background-image": "linear-gradient(90deg, #f1c40f ,  #f8c471)"}),  
                    html.H3(f'{round(float(neut),4)} %'), ], style={'width': '33%', 'float': 'left'}),
        html.Div(children=[html.H3('Negative: ', style={"-webkit-text-fill-color": "transparent",
                    "text-fill-color": "transparent",
                    "-webkit-background-clip": "text",
                    "background-clip": "text",
                    "background-image": "linear-gradient(90deg,  #c0392b,  #ec7063)"}), 
                    html.H3(f'{round(float(neg),4)} %'), ], style={'width': '33%', 'float': 'left'}),
    ])
    return result

PARA_QUE_VAYA_MAS_RAPIDO = True # Hace que vaya más rapido pero salen predicciones aleatorias
WORD2VEC_MODEL       = path.join('MODELS', 'modelo_word2vec_100.model')
CLUSTERING_MODEL     = path.join('MODELS', 'clustering_pipeline.joblib')
MERGER_MODEL_DEFAULT = path.join('MODELS', 'merger_predict_MODEL1.joblib')
FILE_SECTION = 'PDFs'
RAW_DATA_DF = []
SENTIMENT_DATA_DF = []
PREDICTION_PERCENTAGE_DF = []
TIEMPOS = dict()

@app.callback(
    Output('output-section-1', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def procesado_de_pdf(contents, filename):
    import os
    from mergerml import process_PDFs
    
    if not contents:
        return ""

    lugar_de_lectura = os.path.join(FILE_SECTION,filename)
    RAW_DATA, CRONOMETRADO = process_PDFs([lugar_de_lectura], timeit=True)
    TIEMPOS.update(CRONOMETRADO)
    RAW_DATA_DF.append(RAW_DATA)

    salida = html.Div([
        html.Div('Time taken to read data: ' + str(round(TIEMPOS['lectura'],2)) + ' seconds.', style={'margin-bottom':'15px'}),
        html.Div('The annual report was processed successfully.', style={'font-size':'20px','font-weight':'bold'})
        #html.Div('PRIMERAS 10 FRASES PDF:'),
        #html.Div(str(RAW_DATA['sentences'].head(10)), style={'text-align':'justify'})
    ])

    return salida


@app.callback(
    Output('output-section-2', 'children'), # Se actualiza cuando termina la primera lectura 
    Input('output-section-1',  'children'), # de lafunción anterior.
)
def sentimiento_de_frases(contents):
    import os
    from mergerml import get_phrase_sentiment

    if not contents:
        return ""
    
    SENTIMENT_DATA, CRONOMETRADO = get_phrase_sentiment(RAW_DATA_DF[0] if not PARA_QUE_VAYA_MAS_RAPIDO else RAW_DATA_DF[0].sample(500),
                                                        min_words=5,)  # Esto podríamos ponerlo como opción en el Dash
    SENTIMENT_DATA_DF.append(SENTIMENT_DATA)
    TIEMPOS.update(CRONOMETRADO)

    TOP5_NEGATIVAS = SENTIMENT_DATA.sort_values(by='negative', ascending=False).head(10)
    TOP5_NEGATIVAS = TOP5_NEGATIVAS.drop(columns=['length', 'negative','neutral', 'positive'])
    TOP5_NEGATIVAS = TOP5_NEGATIVAS.droplevel(level=[1,2,3]).reset_index().drop(columns=['bank'])
    TOP5_NEGATIVAS = TOP5_NEGATIVAS.rename({'sentences': ''}, axis=1)

    TOP5_POSITIVAS = SENTIMENT_DATA.sort_values(by='positive', ascending=False).head(10)
    TOP5_POSITIVAS = TOP5_POSITIVAS.drop(columns=['length', 'negative','neutral', 'positive'])
    TOP5_POSITIVAS = TOP5_POSITIVAS.droplevel(level=[1,2,3]).reset_index().drop(columns=['bank'])
    TOP5_POSITIVAS = TOP5_POSITIVAS.rename({'sentences': ''}, axis=1)
    OUT_TOP5NEG = [html.Div(f'* {frase}') for frase in TOP5_NEGATIVAS.values[:, 0]]
    OUT_TOP5POS = [html.Div(f'* {frase}') for frase in TOP5_POSITIVAS.values[:, 0]]
    salida = html.Div([
        html.Div(f'Time taken to predict {len(SENTIMENT_DATA)} phrase sentiments: ' + str(round(TIEMPOS['sentiment'],2)) + ' seconds.', style={'margin-bottom':'15px'}),
        html.Div('Financial sentiment analysis has been succesful.', style={'font-size':'20px','font-weight':'bold', 'margin-bottom':'15px'}),
        html.Div('Positive phrases:', style={'margin-bottom':'5px', 'color':'#58d68d'}),
        html.Div(children=OUT_TOP5POS,style={'margin-bottom':'15px','text-align':'justify', 'font-size':'10px'}),
        html.Div('Negative phrases:', style={'margin-bottom': '5px', 'color': '#e74c3c'}),
        html.Div(children=OUT_TOP5NEG, style={'margin-bottom': '15px', 'text-align': 'justify', 'font-size': '10px'}),
    ])

    return salida


@app.callback(
    # Se actualiza cuando termina la primera lectura
    Output('output-section-3', 'children'),
    Input('output-section-2',  'children'),  # de lafunción anterior.
)
def predict_de_merger(contents):
    import os
    from mergerml import predict_merger

    if not contents:
        return ""

    PROBABILIDAD, CRONOMETRADO, SENTIMENT_POR_CLUSTERS = predict_merger(SENTIMENT_DATA_DF[0],
                                                                        WORD2VEC_MODEL,
                                                                        CLUSTERING_MODEL,
                                                                        MERGER_MODEL_DEFAULT, # Se debe seleccionar con un selector
                                                                        timeit=True
                                                                        )
    TIEMPOS.update(CRONOMETRADO)
    SENTIMENT_POR_CLUSTERS['neg'] = SENTIMENT_POR_CLUSTERS['neg'].apply(lambda x: round(x,4))
    SENTIMENT_POR_CLUSTERS['pos'] = SENTIMENT_POR_CLUSTERS['pos'].apply(lambda x: round(x,4))

    salida = html.Div([
        html.Div('Time taken to make prediction: ' + str(round(TIEMPOS['prediction'],2)) + ' seconds.', style={'margin-bottom':'15px', 'margin-left':'10px'}),
        html.Div('Likelihood of M&A:', style={'margin-left':'10px'}),
        html.Div(f'{round(PROBABILIDAD[0]*100,2)}%', style={'font-size':'40px', 'text-align':'center','margin-bottom':'20px', 'margin-left':'10px'}),
        html.Div('Sentiment metrics of each cluster:',  style={'margin-bottom':'15px', 'margin-left':'10px'}),
        dash_table.DataTable(
            SENTIMENT_POR_CLUSTERS.reset_index().rename(columns={'level_2': 'cluster'})[
                ['cluster', 'neg', 'pos']].to_dict('records'),
            [{"name": i, "id": i} for i in ['cluster','neg','pos']]
        )
    ])

    return salida

if __name__ == '__main__':
    app.run_server()
