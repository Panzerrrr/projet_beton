import streamlit as st
import pandas as pd
import numpy as np
import os

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
    st.image(VICAT_LOGO, width=130)
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
    st.image(LE1817_LOGO, width=150)
   


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

############################################# MODEL FOR PREDICT ########################################################
        # Mise en forme pour modifier le titre de la colonne du model.predict
        prediction = model.predict(df.drop(columns=['Strength'],axis=1))
        dfs = pd.DataFrame(prediction, columns = ['Strenght predicted'])
        st.table(dfs[::])

        dfs.style.set_table_styles(
        [{
            'selector': 'th',
            'props': [('background-color', '#add8e6')]
        }])