import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pickle

df = pd.read_csv(r"../../first_exo/concrete_strength_dataset.csv")

def dataSplitting():
    # split into input (X) and output (y) variables
    X = df[['Cement','Blast Furnace Slag','Fly Ash','Water','Superplasticizer','Coarse Aggregate','Fine Aggregate','Age']]
    y = df["Strength"]
    dataSplitting.X_train, dataSplitting.X_test, dataSplitting.y_train, dataSplitting.y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return dataSplitting.X_train, dataSplitting.X_test, dataSplitting.y_train, dataSplitting.y_test
dataSplitting()



def dataStandardisation():

    scaler = StandardScaler()
    dataStandardisation.scaled = scaler.fit_transform(dataSplitting.X_train)
    dataStandardisation.scaled_test = scaler.transform(dataSplitting.X_test)
    return dataStandardisation.scaled ,dataStandardisation.scaled_test
dataStandardisation()



def dataModeling():
    # define the keras model
    dataModeling.model = Sequential()
    dataModeling.model.add(Dense(12, input_dim=8, activation='relu'))
    dataModeling.model.add(Dense(8, activation='relu'))
    dataModeling.model.add(Dense(6, activation='relu'))
    dataModeling.model.add(Dense(1, activation='linear'))
    dataModeling.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    return dataModeling.model
dataModeling()



def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def modelEarlyStopping():
        modelEarlyStopping.stp = EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=0,
                verbose=1,
                mode="auto",
                baseline=None,
                restore_best_weights=False,
        )
        return modelEarlyStopping.stp


def modelFitting():
    modelFitting.history = dataModeling.model.fit(dataSplitting.X_train, dataSplitting.y_train,validation_split=0.30,epochs=1, batch_size=32)
    # ,validation_split=0.33,callbacks=EarlyStopping(monitor='val_loss')
    return modelFitting.history
modelFitting()



def modelLearningCurves():
    # list all data in history
    history = modelLearningCurves.history = modelFitting.history
    plt.figure(figsize=(20,10))
    # summarize history for accuracy
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('model mean_absolute_error')
    plt.ylabel('mean_absolute_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # summarize history for loss
    plt.figure(figsize=(20,10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    return modelLearningCurves.history

modelLearningCurves()


def computeR2Score():
    y_pred = dataModeling.model.predict(dataSplitting.X_test)
    r2_score_value = r2_score(dataSplitting.y_test,y_pred)

    return r2_score_value

def modelDumpedInPickle():
    return pickle.dump(computeR2Score(), open(r'compute_r2_score.pkl', 'wb'))
computeR2Score()