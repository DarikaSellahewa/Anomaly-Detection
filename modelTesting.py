# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 20:31:22 2021

@author: darik
"""

import igraph as ig
import pandas as pd
import preProcess as preProcess
import subgraphMining as subgraph
import GraphViz as graphViz
import NaiveBayes as naiveBayes
import FrequencyCalc as freqCalc
import ColorAssign as colorAssign
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
    
#total dataset
file = pd.read_csv("Data file.csv", delimiter=",")
total_log_event = preProcess.preProcess(file)
total_vertices = subgraph.getVertices(total_log_event)
total_edges = subgraph.getEdges(total_vertices)
master_local_freqTable = freqCalc.getLocalFreqTable(total_vertices, total_edges)
master_freq_table = freqCalc.getGlobalFreqTable(total_vertices, total_edges)

#reading dataset as batch file
test_file = pd.read_csv("Test data.csv", delimiter=",")
log_event = preProcess.preProcess(test_file)
vertices = subgraph.getVertices(log_event)
edges = subgraph.getEdges(vertices)

#loading the model
filename = "Completed_model_g.joblib"

#predicting
anomaly_graph_g, visual_g, freqTable, colors_predicted_g = graphViz.globalVisualisation(vertices,filename)


#getting real colors

#global threshold
high,medium = freqCalc.getGlobalThresholds(master_freq_table)
colors_real_g = colorAssign.getGlobalColours(freqTable, high, medium)


#confusion matrix
colors_real_list_g = list(colors_real_g['Color'])
cm_g = confusion_matrix(colors_real_list_g, colors_predicted_g)



#visualisation
ig.plot(anomaly_graph_g, target= 'graph_result_g.png', **visual_g)


#loading the model
filename = "Completed_model_l.joblib"

#predicting
anomaly_graph_l, visual_l, freqTable, colors_predicted_l = graphViz.globalVisualisation(vertices,filename)


#getting real colors
#local threhold

event_threshold_table, adjFreqTable = freqCalc.getLocalThresholds(master_local_freqTable)

local_freqTable = freqCalc.getLocalFreqTable(vertices, edges)
adjFreqTable_test_data = freqCalc.freqTableAdjust(local_freqTable)
colors_real_l = colorAssign.getLocalColors(adjFreqTable_test_data, event_threshold_table)


#confusion matrix
colors_real_list_l = list(colors_real_l['Color'])
cm_l = confusion_matrix(colors_real_list_l, colors_predicted_l)


#visualisation
ig.plot(anomaly_graph_l, target= 'graph_result_l.png', **visual_l)





#ROC
# =============================================================================
# colors_real = pd.DataFrame(colors_real)
# colors_real[['Event']] = freqTable.index
# colors_real.columns = ['Color']
# =============================================================================
# =============================================================================
# model = naiveBayes.modelTrain(freqTable, colors_real, filename)
# test_data = model.predict_proba(colors_predicted[colors_predicted == 'yellow'])
# fpr, tpr, thresholds = roc_curve(colors_real['Color'].array, colors_predicted, pos_label = 'yellow')
# plt.figure(figsize=(6,4))
# plt.plot(fpr, tpr, linewidth=2)
# plt.plot([0,1], [0,1], 'k--' )
# plt.rcParams['font.size'] = 12
# plt.title('ROC curve for Naive Bayes')
# plt.xlabel('False Positive Rate (1 - Specificity)')
# plt.ylabel('True Positive Rate (Sensitivity)')
# plt.show()
# 
# roc_auc = roc_auc_score(round(freqTable[['Freq_dist']]), round(freqTable[['Freq_dist']]))
# 
# =============================================================================





