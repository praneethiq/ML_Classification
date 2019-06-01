import warnings
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")


def loadCSVData(location):
    return pd.read_csv(location, sep=',', header=None)


def trainTestSplit(xData, yData):
    train = pd.DataFrame()
    test = pd.DataFrame()
    yData.columns = ['class']
    totalData = pd.concat([xData, yData], axis=1)
    unique_classes = yData['class'].unique()
    for index in unique_classes:
        tempData = totalData.loc[totalData['class'] == index]
        tempData = tempData.reset_index(drop=True)
        np.random.seed(123)
        random_indices = np.random.choice(len(tempData), int(round(len(tempData) * 0.8)), replace=False)
        random_non_indices = []
        for i in range(len(tempData)):
            if i not in random_indices:
                random_non_indices += [i]
        train = train.append(tempData.loc[random_indices])
        test = test.append(tempData.loc[random_non_indices])
    xTrain = train.loc[:, train.columns != 'class']
    yTrain = train.loc[:, train.columns == 'class']
    xTest = test.loc[:, test.columns != 'class']
    yTest = test.loc[:, test.columns == 'class']
    yTrain = np.array(yTrain).ravel()
    yTest = np.array(yTest).ravel()
    return xTrain, yTrain, xTest, yTest


def checkMissingValues(inputData):
    counter = 0
    for i in range(len(inputData)):
        for j in range(len(inputData)):
            if (inputData.iloc[i][j] == None):
                print("Missing Value found at(", i, ",", j, ")")
                counter = 1
    if (counter == 0):
        print("No Missing Values are found.")


def featureReduction(xTrain, xTest, xToClassify):

    xTrain = xTrain.iloc[:, 0:255]
    xTest = xTest.iloc[:, 0:255]
    xToClassify = xToClassify.iloc[:, 0:255]
    return xTrain, xTest, xToClassify

def viewFeatureDistribution(xTrain, datasettype):
    mean = np.mean(xTrain.values, dtype=np.float64, axis=0)
    std = np.std(xTrain.values, dtype=np.float64, axis=0)
    fig = plt.figure()
    plt.title("Features Distribution({})".format(datasettype))
    plt.plot(xTrain.columns, np.add(mean, std), color="green")
    plt.plot(xTrain.columns, mean, color="blue")
    plt.plot(xTrain.columns, np.subtract(mean, std), color="green")
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.legend(("Standard Deviation Bounds", "Mean"))


def viewColumnRanges(xTrain, datasettype):
    mean = np.mean(xTrain.values, dtype=np.float64, axis=0)
    min = np.min(xTrain.values, axis=0)
    max = np.max(xTrain.values, axis=0)
    fig = plt.figure()
    plt.title("Input Columns Statistics({})".format(datasettype))
    plt.plot(xTrain.columns, max, color="red")
    plt.plot(xTrain.columns, mean, color="blue")
    plt.plot(xTrain.columns, min, color="orange")
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.legend(('Maximum', 'Average', 'Minimum'))


def GaussianClassifier(xTrain, yTrain, xTest, yTest):

    gnb = GaussianNB()
    model = gnb.fit(xTrain, yTrain)

    print('\nAccuracy of Gaussian classifier on training set: {:.2f}%'
          .format(model.score(xTrain, yTrain)*100))
    print('Accuracy of Guassian classifier on test set: {:.2f}%'
          .format(model.score(xTest, yTest)*100))
    return gnb

def LogisticClassifier(xTrain, yTrain, xTest, yTest):

    logreg = LogisticRegression()
    logreg.fit(xTrain, yTrain)

    print('Accuracy of Logistic regression classifier on training set: {:.2f}%'
          .format(logreg.score(xTrain, yTrain)*100))
    print('Accuracy of Logistic regression classifier on test set: {:.2f}%'
          .format(logreg.score(xTest, yTest)*100))
    return logreg

def DecisionTreeClassification(xTrain, yTrain, xTest, yTest):

    clf = DecisionTreeClassifier().fit(xTrain, yTrain)

    print('Accuracy of Decision Tree classifier on training set: {:.2f}%'
          .format(clf.score(xTrain, yTrain)*100))
    print('Accuracy of Decision Tree classifier on test set: {:.2f}%'
          .format(clf.score(xTest, yTest)*100))
    return clf


def RandomForestClassification(xTrain, yTrain, xTest, yTest):

    RFclf = RandomForestClassifier(n_estimators=10).fit(xTrain, yTrain)

    print('Accuracy of Random Forest classifier on training set: {:.2f}%'
          .format(RFclf.score(xTrain, yTrain)*100))
    print('Accuracy of Random Forest classifier on test set: {:.2f}%'
          .format(RFclf.score(xTest, yTest)*100))
    return RFclf


def predictValues(model, xInput):

    predicted = model.predict(xInput)
    print("\nPredicted Values:")
    print(predicted)
    return predicted


print("\nBeginning of Program Execution:")
print("\nClassifying binary data")
# Load Data from CSV files
xBinData = loadCSVData("X.csv")
yBinData = loadCSVData("y.csv")
xBinToClassify = loadCSVData("XToClassify.csv")



if(len(xBinData) == len(yBinData)):
    print("\nData file loaded Correctly.")
    #Split data into train and test sets
    xTrain, yTrain, xTest, yTest = trainTestSplit(xBinData, yBinData)

    print("Number of Training Sentences =   {}({})%".format(len(xTrain), round(len(xTrain)/len(xBinData)*100)))
    print("Number of Testing Sentences =    {}({})%".format(len(xTest), round(len(xTest)/len(xBinData)*100)))

    # Checking for missing values
    checkMissingValues(xTrain)

    #Checking for each column's range
    viewColumnRanges(xTrain, "Binary")

    #Checking Feature distribution
    viewFeatureDistribution(xTrain, "Binary")
    plt.show()

    #Filtering the extra columns
    xTrain_AllCols, xTest_AllCols = xTrain, xTest
    xTrain, xTest, xBinToClassify = featureReduction(xTrain, xTest, xBinToClassify)

    #Training different models
    gcf = GaussianClassifier(xTrain, yTrain, xTest, yTest)
    lcf = LogisticClassifier(xTrain, yTrain, xTest, yTest)
    dtcf = DecisionTreeClassification(xTrain, yTrain, xTest, yTest)
    rfcf = RandomForestClassification(xTrain, yTrain, xTest, yTest)

    #Predicting based on Random Forest Classifier
    predicted = predictValues(rfcf, xBinToClassify)
    predicted = pd.DataFrame(predicted)
    predicted.to_csv('binaryTask/PredictedClasses.csv', index=False, header=False)

print("\nClassifying multi-class data")
# Load Data from CSV files
xMultiData = loadCSVData("X1.csv")
yMultiData = loadCSVData("y1.csv")
xMultiToClassify = loadCSVData("XToClassify1.csv")



if(len(xMultiData) == len(yMultiData)):
    print("\nData file loaded Correctly.")
    #Split data into train and test sets
    xTrain, yTrain, xTest, yTest = trainTestSplit(xMultiData, yMultiData)

    print("Number of Training Sentences =   {}({})%".format(len(xTrain), round(len(xTrain)/len(xMultiData)*100)))
    print("Number of Testing Sentences =    {}({})%".format(len(xTest), round(len(xTest)/len(xMultiData)*100)))

    # Checking for missing values
    checkMissingValues(xTrain)

    #Checking for each column's range
    viewColumnRanges(xTrain, "Multi-Class")

    #Checking Feature distribution
    viewFeatureDistribution(xTrain, "Multi-Class")
    plt.show()

    #Filtering the extra columns
    xTrain_AllCols, xTest_AllCols = xTrain, xTest
    xTrain, xTest, xMultiToClassify = featureReduction(xTrain, xTest, xMultiToClassify)

    #Training different models
    gcf = GaussianClassifier(xTrain, yTrain, xTest, yTest)
    lcf = LogisticClassifier(xTrain, yTrain, xTest, yTest)
    dtcf = DecisionTreeClassification(xTrain, yTrain, xTest, yTest)
    rfcf = RandomForestClassification(xTrain, yTrain, xTest, yTest)

    #Training based on all the columns before feature reduction
    print("\n\nAccuracies with all the columns before Feature Reduction:")
    gcf_old = GaussianClassifier(xTrain_AllCols, yTrain, xTest_AllCols, yTest)
    lcf_old = LogisticClassifier(xTrain_AllCols, yTrain, xTest_AllCols, yTest)
    dtcf_old = DecisionTreeClassification(xTrain_AllCols, yTrain, xTest_AllCols, yTest)
    rfcf_old = RandomForestClassification(xTrain_AllCols, yTrain, xTest_AllCols, yTest)

    # Predicting based on Random Forest Classifier
    predicted = predictValues(rfcf, xMultiToClassify)
    predicted = pd.DataFrame(predicted)
    predicted.to_csv('multiClassTask/PredictedClasses.csv', index=False, header=False)
