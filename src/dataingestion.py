import pandas as pd
import os
def load_data(input_path,output_path):
    df=pd.read_csv(input_path)
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    df.to_csv(output_path,index=False)
    
if __name__=="__main__":
    load_data("telecom_customer_churn.csv","data/raw/raw.csv") 