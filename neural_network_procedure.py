# Arda Mavi
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

def createNeuralNetwork():
    # Createing Neural Network:
    return MLPClassifier(activation='tanh', hidden_layer_sizes=(8, 2))

def saveNeuralNetwork(clf):
    # Save classifier:
    joblib.dump(clf, 'Data/neural_network/classifier.pkl')

def getNeuralNetwork():
    # Getting trained Neural Network:
    try:
        clf = joblib.load('Data/neural_network/classifier.pkl')
    except:
        return None
    return clf

def trainNeuralNetwork(clf, X, y):
    # Train Neural Network:
    return clf.fit(X, y)

def NeuralNetworkPredict(clf, img):
    # Get predict:
    return clf.predict(img)
