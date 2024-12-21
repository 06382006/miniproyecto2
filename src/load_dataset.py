"""
    Carga del dataset
"""
import pandas as pd


def load_dataset():
    """
    Funcion para leer el dataset de la carpeta raw
    """
    dataset = pd.read_csv("../data/raw/train.csv")
    dataset.to_csv('../data/interim/data_for_eda.csv', index=False)
