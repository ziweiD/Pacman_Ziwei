# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import util
import classificationMethod
import math
import heapq

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """

        "*** YOUR CODE HERE ***"
        countOnTraining = util.Counter()
        countTotalTraining = util.Counter()
        self.prior = util.Counter()

        # scan the data
        for i in range(len(trainingLabels)):
            label = trainingLabels[i]
            self.prior[label] += 1
            data = trainingData[i]
            for feature, f in data.items():
                countTotalTraining[(feature, label)] += 1
                if f > 0:
                    countOnTraining[(feature, label)] += 1

        self.prior.normalize()

        # smoothing
        self.conditionalProb = util.Counter()
        bestAccuracy = 0
        bestK = 0
        bestConditionalP = None
        for k in kgrid:
            conditionalP = util.Counter()

            for label in self.legalLabels:
                for feat in self.features:
                    countOnTraining[ (feat, label)] +=  k
                    countTotalTraining[(feat, label)] +=  2*k

            for x, count in countOnTraining.items():
                conditionalP[x] = count * 1.0 / countTotalTraining[x]
            self.conditionalProb = conditionalP

            guesses = self.classify(validationData)
            accuracy = 0
            for i in range(len(validationLabels)):
                if guesses[i] == validationLabels[i]:
                    accuracy += 1
            accuracy = accuracy * 1.0 / len(validationLabels)
            if accuracy > bestAccuracy or (accuracy == bestAccuracy and k < bestK):
                bestK = k
                bestAccuracy = accuracy
                bestConditionalP = conditionalP

        self.k = bestK
        self.conditionalProb = bestConditionalP

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()

        "*** YOUR CODE HERE ***"
        for label in self.legalLabels:
            logJoint[label] = math.log(self.prior[label])
            for feat, value in datum.items():
                if value > 0:
                    if self.conditionalProb[(feat, label)] == 0:
                        logJoint[label] += 0
                    else:
                        logJoint[label] += math.log(self.conditionalProb[(feat,label)])
                else:
                    logJoint[label] += math.log(1-self.conditionalProb[(feat,label)])

        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featuresOdds = []
        allFeaturesOdds = util.PriorityQueue()

        "*** YOUR CODE HERE ***"
        for feature in self.features:
            odds = self.conditionalProb[(feature, label1)] * 1.0/ self.conditionalProb[(feature, label2)]
            allFeaturesOdds.push(feature, odds)
        count = len(self.features)
        while count > 100:
            allFeaturesOdds.pop()
            count -= 1
        while not allFeaturesOdds.isEmpty():
            f = allFeaturesOdds.pop()
            featuresOdds.append(f)
        return featuresOdds
