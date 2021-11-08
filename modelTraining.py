# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 12:08:07 2021

@author: darik
"""
import igraph as ig
import pandas as pd
import preProcess as preProcess
import subgraphMining as subgraph
import FrequencyCalc as freqCalc
import ColorAssign as colorAssign
import NaiveBayes as naiveBayes
import GraphViz as grpahViz
import neighGraphViz as neighbourGraph
import csv


def prepareData():
    #reading dataset as batch file
    file = pd.read_csv("Data file.csv", delimiter=",")
    training_data = file.sample(frac = 0.75)
    testing_data = file.sample(frac = 0.25)
    training_data.to_csv('Train data.csv')
    testing_data.to_csv('Test data.csv')
    log_event = preProcess.preProcess(training_data)
    vertices = subgraph.getVertices(log_event)
    edges = subgraph.getEdges(vertices)
    
    return vertices, edges

def trainGlobalModel():

    vertices, edges = prepareData()
    #for global frequency calculation
    fre_table = freqCalc.getGlobalFreqTable(vertices, edges)
    #fre_table = freqCalc.getLocalFreqTable(vertices, edges)
    high,medium = freqCalc.getGlobalThresholds(fre_table)
    color_frame = colorAssign.getGlobalColours(fre_table, high, medium)
    graph,visual = grpahViz.graphlVisualisation(fre_table, color_frame['Color'])
    #plotting for data
    ig.plot(graph, target= 'graph_global.png', **visual)
    
    #train naive bayes model- global threshold
    filename = "Completed_model_g.joblib"
    model = naiveBayes.modelTrain(fre_table, color_frame, filename)
    #validation
    accuracy, balanced_acc, weighted_precision, weighted_recall = naiveBayes.validate(fre_table, color_frame, filename)

    return accuracy, balanced_acc, weighted_precision, weighted_recall


def trainLocalModel():
    
    vertices, edges = prepareData()
    
    #for local freq calculation
    Event_fre_table = freqCalc.getLocalFreqTable(vertices, edges)
    #fre_table = fre_table.drop(labels = [8002,8020,8001,4689], axis = 0)
    thresh_table, adjusted_table = freqCalc.getLocalThresholds(Event_fre_table)
    #Frequency table adjustment
    color_frame = colorAssign.getLocalColors(adjusted_table, thresh_table)
    graph,visual = grpahViz.graphlVisualisation(adjusted_table, color_frame['Color'])
    #plotting for data
    ig.plot(graph, target= 'graph_Local.png', **visual)
    
    #train naive bayes model - local thershold
    filename = "Completed_model_l.joblib"
    model = naiveBayes.modelTrain(adjusted_table, color_frame, filename)
    #validation
    accuracy, balanced_acc, weighted_precision, weighted_recall = naiveBayes.validate(adjusted_table, color_frame, filename)
    
    return accuracy, balanced_acc, weighted_precision, weighted_recall



accuracy, balanced_acc, weighted_precision, weighted_recall = trainGlobalModel()
accuracy, balanced_acc, weighted_precision, weighted_recall = trainLocalModel()

#To visualize the neighbor events in the grpah format
vertices, edges = prepareData()
eventFreqTable = freqCalc.getLocalFreqTable(vertices, edges)
#neighbourGraph.getNeighbourGraph(8002, 2, eventFreqTable)












