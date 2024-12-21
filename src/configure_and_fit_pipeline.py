"""
Creación de pipeline para entrneamiento de modelos
"""
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from keras.models import Sequential
from keras.layers import  Dense

def model_fit_pipeline():
    """
    Función para añadir al pipeline el mejor modelo y hacer fit del mismo
    """
    #Carga de train y test
    data_train=pd.read_csv('../data/processed/features_for_model.csv')
    data_test=pd.read_csv('../data/processed/test_dataset.csv')

    #Preparación de data train y test
    #Train
    x_features=data_train.drop(['Default'],axis=1)
    y_target=data_train['Default']
    #Test
    x_features_test=data_test.drop(['Default'],axis=1)
    y_target_test=data_test['Default']

    #Carga del artifact pipeline
    with open ('../Artifacts/pipeline.pkl','rb') as f_var:
        credit_default_model_pipeline=pickle.load(f_var)

    #Aplicación del pipeline al test
    x_features_test_arr=credit_default_model_pipeline.transform(
        x_features_test)
    df_features_test=pd.DataFrame(x_features_test_arr,
                                  columns=x_features_test.columns)

    #Entrenamiento y evaluación de rendimiento de modelos

    #Modelo Naive Bayes
    model_nb = GaussianNB(var_smoothing=0.5)
    model_nb.fit(x_features, y_target)
    y_pred_nb = model_nb.predict(df_features_test)
    acc_nb=accuracy_score(y_target_test,y_pred_nb)

    #Modelo Random Forest
    rf_classifier = RandomForestClassifier(n_estimators=50,
                                           random_state=42,
                                           min_samples_split=10)
    rf_classifier.fit(x_features, y_target)
    y_pred_rf = rf_classifier.predict(df_features_test)
    acc_rf=accuracy_score(y_target_test,y_pred_rf)

    #Decision Tree
    clf = DecisionTreeClassifier(criterion='gini',
                                 max_depth=5,
                                 min_samples_split=5)
    clf.fit(x_features, y_target)
    y_pred_dt = clf.predict(df_features_test)
    acc_dt=accuracy_score(y_target_test,y_pred_dt)

    #Modelo de Regresion
    model_rl = LogisticRegression(C=10,solver='liblinear',
                                  penalty='l2')
    model_rl.fit(x_features, y_target)
    y_pred_rl = model_rl.predict(df_features_test)
    acc_rl=accuracy_score(y_target_test,y_pred_rl)

    #Modelo de Red Neuronal
    model_rn = Sequential([

        Dense(5, activation='relu',
              input_shape=(10,)),
        Dense(1, activation='sigmoid')
    ])
    model_rn.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    model_rn.fit(x_features, y_target, epochs=10)

    y_pred_rn1 = model_rn.predict(df_features_test)
    y_pred_rn = (y_pred_rn1 > 0.5).astype(int)
    acc_rn=accuracy_score(y_target_test,y_pred_rn)


    #Definición de red neuronal para integrar a pipeline (en caso fuera el mejor)
    def create_neural_network():
        """
        Función que define la red 
        neuronal para cargar los steps

        """
        model = Sequential()
        model.add(Dense(5, activation='relu',
                        input_shape=(10,)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model
    #Crear un wrapper para la red neuronal
    class NeuralNetworkStep:
        """
        Steps de la Red neuronal para pipeline
        """
        def __init__(self):
            """Función que guarda el step para llamar a red neuronal
            """
            self.model = create_neural_network()

        def fit(self, x_features, y_target):
            """Función que guarda el step del fit de la red neuronal entrenada

            Args:
                x_features (_type_): data train con ingeniería de características
                y_target (_type_): target a predecir del data train
            """
            self.model.fit(x_features, y_target, epochs=10)

        def predict(self, x_features_test):
            """Función que guarda el step de la predicción

            Args:
                x_features_test (_type_): data test con ingeniería de características

            Returns:
                _type_: retorna la predicción
            """
            return self.model.predict(x_features_test)

    #Crear un diccionario con los modelos y sus precisiones
    modelos = {'nb': acc_nb, 'rf': acc_rf, 'dt': acc_dt, 'rl': acc_rl, 'rn': acc_rn}

    #Encontrar el modelo con la mayor precisión
    mejor_modelo = max(modelos, key=modelos.get)

    #Append para integrar el mejor modelo al pipeline
    if mejor_modelo == "nb":
        credit_default_model_pipeline.steps.append(('modelo_naive_bayes', GaussianNB()))
    elif mejor_modelo == "rf":
        credit_default_model_pipeline.steps.append(('modelo_random_forest',
                                                    RandomForestClassifier()))
    elif mejor_modelo == "dt":
        credit_default_model_pipeline.steps.append(('modelo_decision_tree',
                                                    DecisionTreeClassifier()))
    elif mejor_modelo == "rl":
        credit_default_model_pipeline.steps.append(('modelo_regresion_lineal',
                                                    LogisticRegression()))
    elif mejor_modelo == "rn":
        credit_default_model_pipeline.steps.append(('modelo_red_neuronal',
                                                    NeuralNetworkStep()))

    #Carga y preparación del data train
    train_dataset = pd.read_csv("../data/raw/train.csv")
    train_dataset.drop(['ID'], axis=1, inplace=True)
    train_dataset_features = train_dataset.drop('Default', axis=1)
    train_dataset_target = train_dataset['Default']

    #Fit del pipeline(ya con el modelo incluido) con el data train
    credit_default_model_pipeline.fit(train_dataset_features,train_dataset_target)

    #Almanecamiento del pipeline con modelo incluido
    with open('../artifacts/pipeline_winner_models.pkl','wb') as f_var:
        pickle.dump(credit_default_model_pipeline,f_var)

model_fit_pipeline()
