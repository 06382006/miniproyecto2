"""
Ingenieria de caracteristicas para el modelo final
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def feature_eng():
    """
    funcion para aplicar ingenieria de caracteristicas
    """
    dataset = pd.read_csv('../data/interim/data_for_features.csv')
    # # Ingeniería de Caracatetísticas
    # ## 3.1 Imputación de Variables
    dataset.isnull().mean()

    # Imputamos la media en la columna Mto_ingreso_mensual
    media_ingreso = dataset['Mto_ingreso_mensual'].mean()
    dataset["Mto_ingreso_mensual"] = (
    dataset["Mto_ingreso_mensual"].fillna(media_ingreso).astype(int)
    )
    # Imputamos la media en la columna Nro_dependientes
    # Se redondeo la media para tener numeros enteros
    media_Nro_dependiente = round(dataset['Nro_dependiente'].mean())
    dataset["Nro_dependiente"] = (
    dataset["Nro_dependiente"].fillna(media_Nro_dependiente).astype(int)
    )
    # ### Tratamiento de outliers
    continuas = [
    col
    for col in dataset.columns
    if (dataset[col].dtypes in ["float64", "int64"])
    and (len(dataset[col].unique()) > 30)
    ]
    continuas_outliers=continuas
    for col in continuas_outliers:

        iqr=dataset[col].quantile(0.75)-dataset[col].quantile(0.25)
        iqr
        ll=dataset[col].quantile(0.25)-1.5*iqr
        ul=dataset[col].quantile(0.75)+1.5*iqr
        dataset[col+'_capp']=np.where(dataset[col]>ul,ul,np.where(dataset[col]<ll,ll,dataset[col]))
        print(col)
        plt.figure(figsize=(7,4))
        plt.subplot(121)
        plt.title(col)
        sns.boxplot(y=dataset[col])
        plt.subplot(122)
        plt.title(col + ' outlier')
        sns.boxplot(y=dataset[col+'_capp'])
        plt.show()

    dataset['Prct_uso_tc'] = dataset['Prct_uso_tc_capp']
    dataset['Edad'] = dataset['Edad_capp']
    dataset['Prct_deuda_vs_ingresos'] = dataset['Prct_deuda_vs_ingresos_capp']
    dataset['Mto_ingreso_mensual'] = dataset['Mto_ingreso_mensual_capp']
    dataset['Nro_prod_financieros_deuda'] = dataset['Nro_prod_financieros_deuda_capp']
    dataset=dataset.drop(['Prct_uso_tc_capp'], axis=1)
    dataset=dataset.drop(['Edad_capp'], axis=1)
    dataset=dataset.drop(['Prct_deuda_vs_ingresos_capp'], axis=1)
    dataset=dataset.drop(['Mto_ingreso_mensual_capp'], axis=1)
    dataset=dataset.drop(['Nro_prod_financieros_deuda_capp'], axis=1)

    # # Guaradamos el dataset procesado
    dataset.to_csv('../data/processed/dataset_for_model.csv', index=False)

    # Guardamos de valores de configuracion del train
    feature_eng_configs = {
        'Ing_mensual_imputado' : int(media_ingreso),
        'No_dependiente_imputado' : int(media_Nro_dependiente)
    }

    with open('../artifacts/feature_eng_configs.pkl','wb') as f:
        pickle.dump(feature_eng_configs,f)
