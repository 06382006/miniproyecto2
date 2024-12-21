
# Definir la red neuronal
def create_neural_network():
    model = Sequential()
    model.add(Dense(5, activation='relu', input_shape=(10,)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

class NeuralNetworkStep:
    def __init__(self):
        self.model = create_neural_network()

    def fit(self, X, y):
        self.model.fit(x_features, y_target, epochs=10)

    #def predict(self, X):
        #return self.model.predict(df_features_test)