import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image

################################################# PREDICT #########################################################


from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow import convert_to_tensor
import pandas as pd
import numpy as np
import pickle

############################################# DIR'S PATH ###################################################

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../data/model.h5')
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../data')
VICAT_LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../img/Vicat_SA_logo.svg.png')
LE1817_LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../img/Le1817.jpg')



############################################# LOAD MODEL AND DATA ###################################################


# df = pd.read_csv("../first_exo/concrete_strength_dataset.csv")

model = load_model(MODEL_DIR)
# data = df[0:3:].drop(columns=['Strength'],axis=1)

#############################################LOGO########################################################

# st.write("a logo and text next to eachother")
col1, mid, col2 = st.columns([15,30,20])
with col1:
    st.image(Image.open(VICAT_LOGO), width=130)
with mid:
    html_str_title = f"""
            <style>
            p.a {{
            font: bold 30px Courier;
            }}
            p.t {{
            font-family:sans-serif;
            color: rgb(0,70,133);
            font: bold 50px Times New Roman;
            margin-top : 20px;
            }}

            
            </style>

            <p class="t">NOTEB APP BETON</p>

            """
    # st.write(html_str, width=120)
    st.markdown(html_str_title, unsafe_allow_html=True)
with col2:
    st.image(Image.open(LE1817_LOGO), width=150)
   


############################################# UPLOAD AND SAVE FILE ###################################################

def save_uploadedfile(uploadedfile):
     with open(os.path.join(SAVE_DIR,uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} Loaded !".format(uploadedfile.name))

datafile = st.file_uploader("Upload CSV",type=['csv'])
if datafile is not None:

    file_details = {"FileName":datafile.name,"FileType":datafile.type}
    df  = pd.read_csv(datafile)
    with st.container():
        show_data = st.checkbox("See the raw data?")
        if show_data:
            html_str = f"""
            <style>
            p.a {{
            font: bold 30px Courier;
            }}
            p.b {{
            font-family:sans-serif;
            color: rgb(0,70,133);
            font-size: 24px;
            }}

            
            </style>

            <p class="b">Données à analyser</p>

            """
            st.markdown(html_str, unsafe_allow_html=True)

            st.table(df)
            save_uploadedfile(datafile)
    ############################################## HTML-CSS #######################################
        html_str = f"""
        <style>
        p.a {{
        font: bold 30px Courier;
        }}
        p.b {{
        font-family:sans-serif;
        color: rgb(0,70,133);
        font-size: 24px;
        }}

        
        </style>

        <p class="b">Prédictions du Modéle</p>
        <p class="a"></p>

        """
        st.markdown(html_str, unsafe_allow_html=True)

    ############################################# MODEL FOR PREDICT ###################################
        # Mise en forme pour modifier le titre de la colonne du model.predict
        prediction = model.predict(df.drop(columns=['Strength'],axis=1))
        dfs = pd.DataFrame(prediction, columns = ['Strenght predicted'])
 
        st.table(dfs[::])
    ############################################# SHAPE FOR RESULT'S AND NORMES#########################



    normes = {
    'C8/10': [
        ('Usage décoratif seulement','X0'),
    ], 
    'C12/15': [
        ('Usage décoratif seulement','X0'),
    ], 
    'C16/20': [
        ('Béton de propreté','X0'), 
    ], 
    'C20/25': [
        ('Fondations légères (semelle filante ou isolée','XC1 - XC2'),
    ],
    'C25/30': [
        ('Dallage sur vide sanitaire','XC3 - XC4 - XD1 - XF1 - XF2'), 
    ], 
    'C30/37': [
        ('Dalle/plancher interne à une maison','XD2 - XS1 - XS2 - XF3 - XF4 - XA1'), 
    ], 
    'C35/45': [
        ('Dalle extérieure classique et dallage sur terre-plein, sans contraintes particulières','XD3 - XS3 - XA2'), 
    ], 
    'C45/55': [
        ('Elément soumis à des efforts importants (poutres de très grande portée ou plancher très chargé'), 
    ], 
    'C50/60': [
        ('Béton haute résistance (inutile pour des particuliers et très cher)'), 
    ], 
    'C55/67': [
        ('Béton haute résistance (inutile pour des particuliers et très cher)'), 
    ], 
    'C60/75': [
        ('Béton haute résistance (inutile pour des particuliers et très cher)'), 
    ],
    'C70/85': [
        ('Béton haute résistance (inutile pour des particuliers et très cher)'), 
    ], 
    'C80/95': [
        ('Béton haute résistance (inutile pour des particuliers et très cher)'), 
    ], 
    'C90/105': [
        ('Béton haute résistance (inutile pour des particuliers et très cher)'), 
    ], 
    'C100/115': [
        ('Béton haute résistance (inutile pour des particuliers et très cher)'), 
    ], 
    }


    # st.write(next(iter(normes.values())))
    st.write(list(normes.values()))

    def between_two_values(df_or_dictionnary, norme, start, end):
        matches = {}
        for key, record_list in df_or_dictionnary.items():
            st.write(key,record_list,'record_list')
            for record in record_list:
                value = record
                st.write(record,'record')
                if start < value < end:

                    if key in matches:
                        # st.write(record,'bloop')
                        st.write(matches[key].append(record),'bloop')
                    else:
                        if record > 8 and record < 10:
                            matches[key] = [record]
                            matches[key].insert(len(matches),list(normes.values())[1])
                            # matches[key].append(l[0][0])
                        if record > 12 and record < 15:
                            matches[key] = [record]
                            matches[key].insert(len(matches),list(normes.values())[2])
                            # matches[key].append(l[0][0])
                        if record > 16 and record < 20:
                            matches[key] = [record]
                            matches[key].insert(len(matches),list(normes.values())[3])
                        if record > 20 and record < 25:
                            matches[key] = [record]
                            matches[key].insert(len(matches),list(normes.values())[4])
                            # matches[key].append(l[0][0])
                        if record > 25 and record < 30:
                            matches[key] = [record]
                            matches[key].insert(len(matches),list(normes.values())[5])
                            # matches[key].append(l[0][0])
                        if record > 30 and record < 37:
                            matches[key] = [record]
                            matches[key].insert(len(matches),list(normes.values())[6])
                        if record > 35 and record < 45:
                            matches[key] = [record]
                            matches[key].insert(len(matches),list(normes.values())[7])
                            # matches[key].append(l[0][0])
                        if record > 45 and record < 55:
                            matches[key] = [record]
                            matches[key].append(list(normes.values())[8])
                            # st.write('test')
                        if record > 50 and record < 60:
                            matches[key] = [record]
                            matches[key].insert(len(matches),list(normes.values())[9])
                        if record > 55 and record < 67:
                            matches[key] = [record]
                            matches[key].insert(len(matches),list(normes.values())[9])
                        if record > 60 and record < 75:
                            matches[key] = [record]
                            matches[key].insert(len(matches),list(normes.values())[10])
                        if record > 70 and record < 85:
                            matches[key] = [record]
                            matches[key].insert(len(matches),list(normes.values())[11])
                        if record > 80 and record < 95:
                            matches[key] = [record]
                            matches[key].insert(len(matches),list(normes.values())[12])
                        if record > 90 and record < 105:
                            matches[key] = [record]
                            matches[key].insert(len(matches),list(normes.values())[13])
                        if record > 100 and record < 115:
                            matches[key] = [record]
                            matches[key].insert(len(matches),list(normes.values())[14])

                        
        return matches

    # title = st.text_input('Movie title', 'Life of Brian')
    # st.write('The current movie title is', title)

    col1, col2 = st.columns([15,15])
    with col1:
        # st.image(Image.open(VICAT_LOGO), width=130)
        number = st.number_input('Insert a number',key=1)
        # st.write('The current number is ',width=130,key=1)
    with col2:
        # st.image(Image.open(VICAT_LOGO), width=130)
        number2 = st.number_input('Insert a number',key=2)
        # st.write('The current number is ',width=130,key=2)
        

    

    

    result = between_two_values(dfs,normes, number, number2)
    st.write(result)