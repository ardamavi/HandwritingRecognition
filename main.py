# Arda Mavi
import sys
from os import listdir
from img_procedure import *
from neural_network_procedure import *

def getTrainData():
    # X is training data and y is label:
    X, y = [], []

    # We get the training datasets from training images file:
    train_dir = 'Data/images/train'

    # Geting training characters:
    try:
        char_dirs = listdir(train_dir)
    except:
        return None, None

    # Looking for a character images is not null:
    for char_dir in char_dirs:
        if len(listdir(train_dir+'/'+char_dir)) < 1:
            char_dirs.remove(char_dir)
        else:
            continue
    # If all character lists is null, return None:
    if len(char_dirs) < 1:
        return None, None
    # Getting image datasets from all character dir:
    for char_dir in char_dirs:
        for img_dir in listdir(train_dir+'/'+char_dir):
            X.append([getImg(train_dir+'/'+char_dir+'/'+img_dir)])
            y.append(ord(char_dir))

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
            print('Training datasets not found!')
            return
        # TODO: Make regular X data
        clf = trainNeuralNetwork(clf, X, y)
        saveClf(clf)

    # Predict:
    print('Predict:', chr(NeuralNetworkPredict(clf, img)))

    # Finish main:
    return


if __name__ == '__main__':
    main()
