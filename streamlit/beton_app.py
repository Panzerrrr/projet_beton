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

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..\data\model.h5')
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..\data')



############################################# LOAD MODEL AND DATA ###################################################


# df = pd.read_csv("../first_exo/concrete_strength_dataset.csv")

model = load_model(MODEL_DIR)
# data = df[0:3:].drop(columns=['Strength'],axis=1)



############################################# UPLOAD AND SAVE FILE ###################################################

def save_uploadedfile(uploadedfile):
     with open(os.path.join(SAVE_DIR,uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to df Dir".format(uploadedfile.name))

datafile = st.file_uploader("Upload CSV",type=['csv'])
if datafile is not None:
   file_details = {"FileName":datafile.name,"FileType":datafile.type}
   df  = pd.read_csv(datafile)
   st.dataframe(df)
   save_uploadedfile(datafile)
   ##### USE DATA'S SAVED WITH THE MODEL FOR PREDICT #####
   st.write(model.predict(df.drop(columns=['Strength'],axis=1)))
   


