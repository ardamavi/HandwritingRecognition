# Arda Mavi
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

def createNeuralNetwork():
    # Createing Neural Network:
    return MLPClassifier(hidden_layer_sizes=(40, 28, 10))

def getNeuralNetwork():
    # Getting trained Neural Network:
    try:
        clf = joblib.load('Data/neural_network/classifier.pkl')
    except:
        return None
    return clf

def trainNeuralNetwork(clf, X, y):
    # Training Neural Network:
    return clf.fit(X, y)

def getScore(clf, X, y):
    return clf.score(X, y)

def getPredict(clf, img):
    # Get predict:
    return clf.predict(img)

def saveNeuralNetwork(clf):
    # Save classifier:
    joblib.dump(clf, 'Data/neural_network/classifier.pkl')
