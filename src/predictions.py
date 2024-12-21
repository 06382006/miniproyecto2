"""
Prediccion con nuevos valores
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def prediccion_new_data():
    """
    funcion para prediccion de dataset nuevo
    """
    dataset_test = pd.read_csv('../data/raw/test.csv')

    # Proceso de preparacion de datos para predecir
    with open('../artifacts/feature_eng_configs.pkl','rb') as f:
        feature_eng_config = pickle.load(f)
    #Removemos la columna ID
    dataset_test.drop(['ID'], axis=1, inplace=True)

    # Imputamos Mto Ingreso Mensual
    dataset_test["Mto_ingreso_mensual"] = dataset_test["Mto_ingreso_mensual"].fillna(
    feature_eng_config["Ing_mensual_imputado"]
    )

    # Imputamos Nro dependientes
    dataset_test["Nro_dependiente"] = dataset_test["Nro_dependiente"].fillna(
    feature_eng_config["No_dependiente_imputado"]
    )

    # Tratamiento de outliers
    continuas = [
    col
    for col in dataset_test.columns
    if (dataset_test[col].dtypes in ["float64", "int64"])
    and (len(dataset_test[col].unique()) > 30)
    ]
    continuas_outliers=continuas
    for col in continuas_outliers:
        iqr=dataset_test[col].quantile(0.75)-dataset_test[col].quantile(0.25)
        ll=dataset_test[col].quantile(0.25)-1.5*iqr
        ul=dataset_test[col].quantile(0.75)+1.5*iqr
        dataset_test[col + "_capp"] = np.where(
        dataset_test[col] > ul, ul, np.where(dataset_test[col] < ll, ll, dataset_test[col])
        )

        print(col)
        plt.figure(figsize=(7,4))
        plt.subplot(121)
        plt.title(col)
        sns.boxplot(y=dataset_test[col])
        plt.subplot(122)
        plt.title(col + ' outlier')
        sns.boxplot(y=dataset_test[col+'_capp'])
        plt.show()

    dataset_test['Prct_uso_tc'] = dataset_test['Prct_uso_tc_capp']
    dataset_test['Edad'] = dataset_test['Edad_capp']
    dataset_test['Prct_deuda_vs_ingresos'] = dataset_test['Prct_deuda_vs_ingresos_capp']
    dataset_test['Mto_ingreso_mensual'] = dataset_test['Mto_ingreso_mensual_capp']
    dataset_test['Nro_prod_financieros_deuda'] = dataset_test['Nro_prod_financieros_deuda_capp']
    dataset_test=dataset_test.drop(['Prct_uso_tc_capp'], axis=1)
    dataset_test=dataset_test.drop(['Edad_capp'], axis=1)
    dataset_test=dataset_test.drop(['Prct_deuda_vs_ingresos_capp'], axis=1)
    dataset_test=dataset_test.drop(['Mto_ingreso_mensual_capp'], axis=1)
    dataset_test=dataset_test.drop(['Nro_prod_financieros_deuda_capp'], axis=1)

    # Estandarizacion con objeto scaler de train
    with open('../artifacts/std_scaler.pkl','rb') as f:
        std_scaler = pickle.load(f)

    # Cargamos modelo entranado
    with open('../models/random_forest_v1.pkl','rb') as f:
        model = pickle.load(f)

    X_data_test_std = std_scaler.transform(dataset_test)
