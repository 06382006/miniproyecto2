"""
Analisis exploratorio
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def exploratory_analysis():
    """
    funcion para realizar el analisis exploratorio
            """

    dataset = pd.read_csv('../data/interim/data_for_eda.csv')

    # # Eliminamos variables no utiles
    dataset.drop(['ID'], axis=1, inplace=True)

    # # Analisis estadistico descriptivo de las variables
    dataset.describe()

    # # Graficos boxplot para identificar variables clasificatorias
    for col in dataset.columns:
        plt.figure(figsize=(7,4))

        plt.title(col)
        sns.boxplot(x=dataset['Default'],y=dataset[col])
        plt.show()

    # # Guardamos dataset final para ingenieria de caracteristicas
    dataset.to_csv('../data/interim/data_for_features.csv', index=False)
