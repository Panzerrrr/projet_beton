import streamlit as st
import pandas as pd
import numpy as np
import os

########### PREDICT ###########


from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow import convert_to_tensor
import pandas as pd
import numpy as np
from PIL import Image


########### DIR'S PATH ###########

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../data/model.h5')
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../data')

########### LOAD MODEL AND DATA ###########

model = load_model(MODEL_DIR)
image = Image.open('Preditbet.png')

########### CUSTOMIZING PAGE TITLE AND FAVICON ###########

st.set_page_config(page_title='PrediBet', page_icon=Image.open('Favicon.png'))

########### HIDE BURGER MENU ###########

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
span {color: #1F2023;}
</style> """, unsafe_allow_html=True)

########### CONDENSE LAYOUT ###########

padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

########### BETON NORM LIST ###########

normes = {
    '8': ['Usage décoratif seulement','X0'], 
    '16': ['Béton de propreté','X'], 
    '20': ['Fondations légères','XC1 - XC2'],
    '25': ['Dallage sur vide sanitaire','XC3 - XC4 - XD1 - XF1 - XF2'], 
    '30': ['Dalle/plancher interne à une maison','XD2 - XS1 - XS2 - XF3 - XF4 - XA1'], 
    '35': ['Dalle classique ou sur terre-plein','XD3 - XS3 - XA2'], 
    '45': ['Elément soumis à des efforts importants','Haute Résistance'], 
    '50': ['Béton haute résistance','Haute Résistance'], 
    }

########### TITLE INTRO ###########

# st.title('PrediBet App')
# colimg1, colimg2, colimg3 = st.columns([2,4,2])

# with colimg1:
#     st.write("")

# with colimg2:
#      st.image(image)

# with colimg3:
#     st.write("")

st.image(image)

st.subheader('Déterminer la force du béton grâce à l\'intelligence artificielle')

########### UPLOAD AND SAVE FILE ###########

st.subheader('Veuillez charger un document compatible (format csv)')

def save_uploadedfile(uploadedfile):
     with open(os.path.join(SAVE_DIR,uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Force du béton prédite !")

########### CHECK NORM FOR SPECIFIC PREDICTION ###########
def check_norm(predicted_score):
    norm_list = []
    # return predicted_score[0]
    for l in predicted_score:
        # return(l[0])
        recipe = {
            'prediction':l[0],
            'description':'',
            'norm':''
        }
        if l[0] <= 15:
            print('15')
            recipe['description'] = normes['8'][0]
            recipe['norm'] = normes['8'][1]
            norm_list.append(recipe)
        elif l[0] <= 20:
            print('20')
            recipe['description'] = normes['16'][0]
            recipe['norm'] = normes['16'][1]
            norm_list.append(recipe)
        elif l[0] <= 25:
            print('25')
            recipe['description'] = normes['20'][0]
            recipe['norm'] = normes['20'][1]
            norm_list.append(recipe)
        elif l[0] <= 30:
            print('30')
            recipe['description'] = normes['25'][0]
            recipe['norm'] = normes['25'][1]
            norm_list.append(recipe)
        elif l[0] <= 35:
            print('35')
            recipe['description'] = normes['30'][0]
            recipe['norm'] = normes['30'][1]
            norm_list.append(recipe)
        elif l[0] <= 45:
            print('45')
            recipe['description'] = normes['35'][0]
            recipe['norm'] = normes['35'][1]
            norm_list.append(recipe)
        elif l[0] <= 50:
            print('50')
            recipe['description'] = normes['45'][0]
            recipe['norm'] = normes['45'][1]
            norm_list.append(recipe)
        else:
            print('115')
            recipe['description'] = normes['50'][0]
            recipe['norm'] = normes['50'][1]
            norm_list.append(recipe)
    return norm_list

datafile = st.file_uploader("Fichier CSV nécessaire",type=['csv'])
if datafile is not None:
#    col1, col2, col3 = st.columns([2,2,2])
   file_details = {"FileName":datafile.name,"FileType":datafile.type}
   df = pd.read_csv(datafile)
#    col1.subheader("Fichier Chargé")
#    col1.dataframe(df)
   save_uploadedfile(datafile)

   ##### USE DATA'S SAVED WITH THE MODEL FOR PREDICT #####
#    col2.subheader("Force Prédite")
   predictions = model.predict(df.drop(columns=['Strength'],axis=1))
#    col2.write(predictions)
   norm_list = check_norm(predictions)
#    col3.write(norm_list)

   metric_col1, metric_col2, metric_col3 = st.columns([0.2,0.6,1]) 

   i = 1 
   for list in norm_list:
        metric_col1.metric('Recette', i)
        metric_col2.metric(list['description'], 'résistance: ' + str(int(list['prediction'])))
        metric_col3.metric("Classe d'exposition", list['norm'])
        i += 1


