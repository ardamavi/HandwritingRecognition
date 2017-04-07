# Arda Mavi
import sys
from os import listdir
from img_procedure import *
from neural_network_procedure import *

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

def main():
    # Get image directory:
    # We get the arguman from run commend:
    imgDir = getImgDir(sys.argv)
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
