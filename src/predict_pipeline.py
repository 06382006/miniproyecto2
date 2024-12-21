"""contiene el código que carga el pipeline ya entrenado
    y recibe los datos de testing
"""
import pickle
import pandas as pd

def predict_model():
    """función que ejecuta el pipeline para la predicción del test con 
    el modelo ganador
    """
    #Carga del pipeline con el modelo ganador
    with open ('../Artifacts/pipeline_winner_models.pkl','rb') as f_var:
        credit_default_model_pipeline=pickle.load(f_var)

    #Carga del dataset con los datos del testing para predicción
    test_dataset = pd.read_csv("../data/raw/test.csv")
    test_dataset.drop(['ID'], axis=1, inplace=True)

    #Aplicación del pipeline para predicción y resultados
    rn_prediccion = credit_default_model_pipeline.predict(test_dataset)
    rn_prediccion = (rn_prediccion > 0.5).astype(int)
    
predict_model()