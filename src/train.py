import os
import yaml
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from keras.model import Sequential

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
        callback=keras.callback.EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True,start_from_epoch=10)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="binary_crossentropy",
        callback=[callback],
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
    model=build_model()
    model.fit(X_train,y_train,batch_size=params["batch_size"],epochs=params["epochs"],validation_step=0.2,verbose=1)
    
    #saving the model
    os.makedirs(os.path.dirname(model_paths),exist_ok=True)
    model.save(model_paths)
    
    
if __name__=="__main":
    train(data_dir="data/processed",
        model_path="models/model.keras",
        params_path="params.yaml")