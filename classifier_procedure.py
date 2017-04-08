# Arda Mavi
from sklearn.externals import joblib
from sklearn import tree

def createDecisionTree():
    # Createing classifier:
    return tree.DecisionTreeClassifier()

def getClassifier():
    # Getting trained classifier:
    try:
        clf = joblib.load('Data/classifier/classifier.pkl')
    except:
        return None
    return clf

def trainClassifier(clf, X, y):
    # Training classifier:
    return clf.fit(X, y)

def getScore(clf, X, y):
    return clf.score(X, y)

def getPredict(clf, img):
    # Get predict:
    return clf.predict(img)

def saveClassifier(clf):
    # Save classifier:
    joblib.dump(clf, 'Data/classifier/classifier.pkl')
