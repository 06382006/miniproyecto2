"""
Creación de pipeline de ingeniería de características
"""
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_engine.imputation import MeanMedianImputer
from feature_engine.outliers import Winsorizer

def pipeline_features():
    """
    Función para crear el pipeline de ingeniería de características para el dataset
    del modelo
    """
    #Lectura del dataset
    dataset = pd.read_csv("../data/raw/train.csv")

    #Deficinción de variables para ingeniería de características
    target='Default'
    vars_to_drop=['ID',target]
    continue_vars_to_imputation=['Mto_ingreso_mensual','Nro_dependiente']
    continue_vars_outliers=['Prct_uso_tc','Edad','Prct_deuda_vs_ingresos',
        'Mto_ingreso_mensual','Nro_prod_financieros_deuda']

    #Separación de variables en train y test
    x_features=dataset.drop(labels=vars_to_drop,axis=1)
    y_target=dataset[target]
    x_train,x_test,y_train,y_test=train_test_split(
        x_features,y_target, test_size=0.3,shuffle=True,random_state=2025)

    #Creación de pipeline
    credit_default_predict_model=Pipeline([

        #Imputación de variables continuas
        ('continue_var_mean_imputation',MeanMedianImputer(imputation_method='mean',
        variables=continue_vars_to_imputation)),

        #Tratamiento de outliers
        ('continue_outliers_treatment',Winsorizer(
            capping_method="iqr",variables=continue_vars_outliers)),

        #Estandarización de variables
        ('feature_scaling',StandardScaler())
    ])

    #Fit del pipeline con los datos del train
    credit_default_predict_model.fit(x_train)

    #Creación del xtrain y xtest luego de ejecutar pipeline
    x_features_processed=credit_default_predict_model.transform(x_train)
    df_features_processed=pd.DataFrame(x_features_processed,columns=x_train.columns)
    df_features_processed[target]=y_train.reset_index()['Default']
    x_test[target]=y_test

    #Almacenamiento del train y test en carpetas y pipeline pkl
    df_features_processed.to_csv('../data/processed/features_for_model.csv',index=False)
    x_test.to_csv('../data/processed/test_dataset.csv',index=False)

    with open('../artifacts/pipeline.pkl','wb') as f_var:
        pickle.dump(credit_default_predict_model,f_var)

pipeline_features()
        