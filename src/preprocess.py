import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def preprocess(input_path,output_dir):
    df=pd.read_csv(input_path)
    df=df[[
        'Gender', 'Age', 'Married', 'Number of Dependents',
        'City', 'Zip Code', 
        'Latitude', 'Longitude', 'Number of Referrals',
        'Tenure in Months', 'Offer', 'Phone Service',
        'Avg Monthly Long Distance Charges', 'Multiple Lines',
        'Internet Service', 'Internet Type', 'Avg Monthly GB Download',
        'Online Security', 'Online Backup', 'Device Protection Plan',
        'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
        'Streaming Music', 'Unlimited Data', 'Contract', 'Paperless Billing',
        'Payment Method', 'Monthly Charge', 'Total Charges', 'Total Refunds',
        'Total Extra Data Charges', 'Total Long Distance Charges',
        'Total Revenue', 'Customer Status', 'Churn Category', 'Churn Reason']]
    
    internet_cat_cols=['Internet Type','Online Security','Online Backup','Device Protection Plan',
       'Premium Tech Support','Streaming TV','Streaming Movies',
       'Streaming Music','Unlimited Data']
    internet_num_cols=['Avg Monthly GB Download']
    phone_cat_cols=['Multiple Lines']
    phone_num_cols=[ 'Avg Monthly Long Distance Charges']
    offers_cols=["Offer"]
    impute_cat_cols=SimpleImputer(strategy="constant",fill_value="None")
    impute_num_cols=SimpleImputer(strategy="constant",fill_value=0)

    preprocessing_1=ColumnTransformer(transformers=[
        ( "int_cat",impute_cat_cols,internet_cat_cols),
            ("int_num",impute_num_cols,internet_num_cols),
            ("offer_col",impute_cat_cols,offers_cols),
            ("ph_cat",impute_cat_cols,phone_cat_cols),
            ("ph_num",impute_num_cols,phone_num_cols)
            
        ],
        remainder="passthrough")

    preprocessing_1.set_output(transform="pandas")
    df_new=preprocessing_1.fit_transform(df)
    df_new.drop(columns=['remainder__Churn Category','remainder__Churn Reason'],axis=1,inplace=True)
    ord_categ=["Month-to-Month","One Year","Two Year"]
    ord_enc=OrdinalEncoder(categories=[ord_categ])
    df_new["remainder__Contract"]=ord_enc.fit_transform(df_new[['remainder__Contract']])
    df_new=df_new[df_new['remainder__Customer Status']!="Joined"].copy()
    X=df_new.drop("remainder__Customer Status",axis=1)
    y=df_new['remainder__Customer Status']
    df_cat_cols=X.select_dtypes(include=["object"]).columns
    df_num_cols=X.select_dtypes(include=['int64',"float64"]).columns
    one_hot=OneHotEncoder(drop='first', sparse_output=False)
    std_scale=StandardScaler()
    preprocessing2=ColumnTransformer(transformers=[
        ("cat_cols",one_hot,df_cat_cols),
        ("num_cols",std_scale,df_num_cols)
    ])
    preprocessing2.set_output(transform="pandas")
    X_prep=preprocessing2.fit_transform(X)
    maps={"Churned":1,"Stayed" :0}
    y=y.map(maps)
    X_train,X_test,y_train,y_test=train_test_split(X_prep,y,test_size=0.2,random_state=42,stratify=y)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    joblib.dump(preprocessing2, "models/preprocessor.pkl")

    print("Preprocessing complete ✔")


if __name__ == "__main__":
    preprocess(
        r"data\raw\telecom_customer_churn.csv",
        r"data/processed"
    )