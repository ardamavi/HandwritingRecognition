# Arda Mavi
import sys
from os import listdir
from img_procedure import *
from classifier_procedure import *

def getDatasetsFromDir(datasets_dir):
    # This function return training or tasting input(X: Image matris) and output(y: Decimal character) datasets from all directory on datasets_dir
    # X is training data and y is label:
    X_train, y_train, X_test, y_test = [], [], [], []

    # Geting training characters:
    try:
        char_dirs = listdir(datasets_dir)
    except:
        return None, None

    # '.DS_Store' is not a character directory, delete it:
    if '.DS_Store' in char_dirs:
        char_dirs.remove('.DS_Store')

    # Looking for a character images is not empty:
    for char_dir in char_dirs:
        if len(listdir(datasets_dir+'/'+char_dir)) < 1:
            char_dirs.remove(char_dir)
        else:
            continue
    # If all character lists is empty, return None:
    if len(char_dirs) < 1:
        return None, None

    try:
        # Adding empty images to train dataset:
        emptyBlack = getImg('Data/images/emptyBlack.png')
        X_train.append(getImg('Data/images/emptyWhite.png'))
        X_train.append(getImg('Data/images/emptyBlack.png'))
        y_train.append(ord(' '))
        y_train.append(ord(' '))
    except:
        print('Empty images(for training) not found!')

    # Getting image datasets from all character dir:
    for char_dir in char_dirs:
        img_dirs = listdir(datasets_dir+'/'+char_dir)

        # '.DS_Store' is not a image directory, delete it:
        if '.DS_Store' in img_dirs:
            img_dirs.remove('.DS_Store')

        # Split the traning and test images:
        point = int(0.9*len(img_dirs))
        train_imgs, test_imgs = img_dirs[:point], img_dirs[point:]

        # Createing train and test image matrix list:
        for img_dir in train_imgs:
            X_train.append(getImg(datasets_dir+'/'+char_dir+'/'+img_dir))
            y_train.append(ord(char_dir))
        for img_dir in test_imgs:
            X_test.append(getImg(datasets_dir+'/'+char_dir+'/'+img_dir))
            y_test.append(ord(char_dir))

    return X_train, y_train, X_test, y_test

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

    # Get saved classifier:
    clf = getClassifier()

    # If you have not saved classifier already, create new:
    if clf is None:
        clf = createDecisionTree()
        # We get the training datasets from training images file:
        X_train, y_train, X_test, y_test = getDatasetsFromDir('Data/images/train')
        if X_train is None:
            print('Characters datasets not found!')
            return

        # Training classifier
        clf = trainClassifier(clf, X_train, y_train)

        # Saveing classifier:
        saveClassifier(clf)

        # Scores:
        print('Train Score:', getScore(clf, X_train, y_train))
        print('Test Score:', getScore(clf, X_test, y_test))

    # Predict:
    print('Predict:', chr(getPredict(clf, img)))

    # Finish main:
    return


if __name__ == '__main__':
    main()
