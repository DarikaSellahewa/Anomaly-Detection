# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 19:38:17 2021

@author: darik
"""

import pandas as pd
import ColorAssign as colorAssign
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
import joblib

class Model:
    
    naiveBayesClassifier = make_pipeline( DictVectorizer(), MultinomialNB())
    
    def setModel(self, model, filename): 
        joblib.dump(model, filename)
        
    def getModel(self, filename):
        model = joblib.load(filename)
        return model
    


def createTrainingSet(freqTable):
    
    class_table = freqTable.sort_values(by = ['Event'])
    
    #replace index column
    class_table.reset_index(drop=True, inplace=True)
    events = freqTable.index
    
    #adding event column
    class_table[['Event']] = list(events.sort_values(ascending = True))
    
    return class_table

def modelTrain(freqTable, colorFrame, filename):
    
    class_table = createTrainingSet(freqTable)
    
    #adding color column
    colorframe = colorFrame.sort_values(by = ['Event'])
    #replace index column
    colorframe.reset_index(drop=True, inplace=True)
    class_table = pd.concat([class_table, colorframe], axis=1 )
    
    class_dict = class_table[["Event","Freq_dist"]].to_dict('records')
    class_target = class_table['Color']
    
    model =  Model()
    naiveBayesClassifier = make_pipeline( DictVectorizer(), MultinomialNB())
    naiveBayesClassifier.fit(class_dict, class_target)
    model.setModel(naiveBayesClassifier, filename)
    
    return naiveBayesClassifier

    
def getPrediction(freqTable, filename):
    
    dataset = createTrainingSet(freqTable)
    test_dict = dataset[["Event","Freq_dist"]].to_dict('records')
    
    #Predicting
    model = Model()
    nbModel = model.getModel(filename)
    labels = nbModel.predict(test_dict)
    
    return labels
    
def getAccuracy(freqTable, colorFrame, filename):
    
    class_table = createTrainingSet(freqTable)
    #adding color column
    colorframe = colorFrame.sort_values(by = ['Event'])
    #replace index column
    colorframe.reset_index(drop=True, inplace=True)
    class_table = pd.concat([class_table, colorframe], axis=1 )
    
    class_dict = class_table[["Event","Freq_dist"]].to_dict('records')
    class_target = class_table['Color']
    
    model =  Model()
    accuracy = cross_val_score(model.getModel(filename), class_dict, class_target, cv=10, scoring="accuracy")
    balanced_acc = cross_val_score(model.getModel(filename), class_dict, class_target, cv=10, scoring="balanced_accuracy")
    weighted_precision = cross_val_score(model.getModel(filename), class_dict, class_target, cv=10, scoring="precision_weighted")
    weighted_recall = cross_val_score(model.getModel(filename), class_dict, class_target, cv=10, scoring="recall_weighted")

    accuracy = accuracy.mean()*100
    balanced_acc = balanced_acc.mean()*100
    weighted_precision = weighted_precision.mean()*100
    weighted_recall = weighted_recall.mean()*100
    
    return accuracy, balanced_acc, weighted_precision, weighted_recall
    