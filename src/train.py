import os
import yaml
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from keras import Sequential

def build_model(input_dim, params):
        model=keras.Sequential([
        keras.layers.Dense(params["units_1"], activation="relu", input_shape=(input_dim,)),
        keras.layers.Dropout(params["dropout"]),
        keras.layers.Dense(params["units_2"], activation="relu"),
        keras.layers.Dropout(params["dropout"]),
        keras.layers.Dense(params["units_3"], activation="relu"),
        keras.layers.Dropout(params["dropout"]),
        keras.layers.Dense(1, activation="sigmoid")
        ]
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"])
        
        return model
    
def train(data_dir,model_paths,param_paths):
    #loading params 
    with open( param_paths) as f:
        params = yaml.safe_load(f)["train"]
    #loading the data 
    X_train=pd.read_csv(f"{data_dir}/X_train.csv")
    y_train=pd.read_csv(f"{data_dir}/y_train.csv")
    #Building model
    model=build_model(X_train.shape[1],params)
    callback=keras.callbacks.EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True,start_from_epoch=10)
    model.fit(X_train,y_train,batch_size=params["batch_size"],epochs=params["epochs"],validation_split=0.2,verbose=1,callbacks=[callback])
    
    #saving the model
    os.makedirs(os.path.dirname(model_paths),exist_ok=True)
    model.save(model_paths)
    
    
if __name__=="__main__":
        train(data_dir="data/processed",
            model_paths="models/model.keras",
            param_paths="param.yaml")