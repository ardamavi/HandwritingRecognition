import sys
import numpy
from PIL import Image
from os import listdir
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

def getImg(img_dir):
    try:
        # Get grayscale image matrix from directory:
        img = Image.open(img_dir).convert('L')
    except:
        return None
    # If image size not 25x25, resize:
    if img.size != (25, 25):
        img = img.resize((25, 25))
    # Return image matrix:
    return numpy.array(img)

def getImgDir():
    # We get the arguman from run commend:
    argumans = sys.argv
    if len(argumans) < 2:
        return None
    return argumans[1]

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

def getTrainData():
    X, y = [], []
    train_dir = 'Data/images/train'
    try:
        data0Imgs = listdir(train_dir+'/0')
        data1Imgs = listdir(train_dir+'/1')
    except:
        return None, None
    if len(data0Imgs) < 1 or len(data1Imgs) < 1:
        return None, None

    for img_dir in data0Imgs:
        X.append([getImg(train_dir+'/0/'+img_dir)])
        y.append(0)

    for img_dir in data1Imgs:
        X.append([getImg(train_dir+'/1/'+img_dir)])
        y.append(1)

    return X, y

def NeuralNetworkPredict(clf, img):
    # Get predict:
    return clf.predict(img)

def main():
    # Get image directory:
    imgDir = getImgDir()
    if imgDir is None:
        print('Image directory not found!')
        return

    # Get image from image directory:
    img = getImg(imgDir)
    if img is None:
        print('Image not found!')
        return

    # Get saved Neural Network:
    clf = getNeuralNetwork()

    # If you have not saved Neural Network already, create new:
    if clf is None:
        clf = createNeuralNetwork()
        X, y = getTrainData()
        if X is None:
            print('Train data not found!')
            return
        # TODO: Make regular X data
        clf = trainNeuralNetwork(clf, X, y)
        saveClf(clf)

    # Predict:
    print('Predict:', NeuralNetworkPredict(clf, img))

    # Finish main:
    return


if __name__ == '__main__':
    main()
