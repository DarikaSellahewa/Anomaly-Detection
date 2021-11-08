# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 01:00:07 2021

@author: darik
"""

from sklearn.svm import OneClassSVM
from numpy import quantile, where
import matplotlib.pyplot as plt
import pandas as pd
import preProcess as preProcess
import subgraphMining as subgraph
import FrequencyCalc as freqCalc
import ColorAssign as colorAssign
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
import joblib




class SVMModel:
    
    
    def setModel(self, model, filename): 
        joblib.dump(model, filename)
        
    def getModel(self, filename):
        model = joblib.load(filename)
        return model
    

def dataprep():
    #reading dataset as batch file
    file = pd.read_csv("Test data.csv", delimiter=",")
    log_event = preProcess.preProcess(file)
    vertices = subgraph.getVertices(log_event)
    edges = subgraph.getEdges(vertices)
    fre_table = freqCalc.getGlobalFreqTable(vertices, edges)
    data = fre_table[['Scaled_Freq']]
    data = data.sort_index()
    events = data.index
    data.reset_index(drop=True, inplace=True)
    data[['Event']] = list(events.sort_values(ascending = True))

    return fre_table, data, vertices, edges


def svmFitPredict(data, filename):
    #model
    svm = OneClassSVM(cache_size=200, coef0=0.0, degree=3, gamma=0.001, kernel='rbf',
            max_iter=-1, nu=0.03, shrinking=True, tol=0.001, verbose=False) 
    svmModel =  SVMModel()
    svmModel.setModel(svm, filename)
    pred = svm.fit_predict(data)
    
    return pred

def crossValidate(data, predicted_result, filename):
    
    svMModel = SVMModel()
    svm = svMModel.getModel(filename)
    
    balanced_accuracy = cross_val_score(svm, data, predicted_result, cv=10, scoring="balanced_accuracy")
    weighted_precision = cross_val_score(svm, data, predicted_result, cv=10, scoring="precision_weighted")
    weighted_recall = cross_val_score(svm, data, predicted_result, cv=10, scoring="recall_weighted")
    f_score = cross_val_score(svm, data, predicted_result, cv=10, scoring="f1_weighted")
    
    balanced_accuracy = balanced_accuracy.mean()*100
    weighted_precision = weighted_precision.mean()*100
    weighted_recall = weighted_recall.mean()*100
    f_score = f_score.mean()*100
    
    return balanced_accuracy,weighted_precision,weighted_recall,f_score 


def getBaselineGlobalAnomalies(data):
    #global threshold model
    high,medium = freqCalc.getGlobalThresholds(data)
    colors_real = colorAssign.getGlobalColours(data, high, medium)
    colors_real[["Color"]] = colors_real[["Color"]].replace(["green"],["red"])
    colors_real[["Color"]] = colors_real[["Color"]].replace(["yellow","red"],[1,-1])

    return colors_real

def getBaselineLocalAnomalies(vertices, edges):
    #local threshold model
    local_freqTable = freqCalc.getLocalFreqTable(vertices, edges)
    event_threshold_table, adjFreqTable = freqCalc.getLocalThresholds(local_freqTable)
    colors_real = colorAssign.getLocalColors(adjFreqTable, event_threshold_table)
    colors_real[["Color"]] = colors_real[["Color"]].replace(["green"],["red"])
    colors_real[["Color"]] = colors_real[["Color"]].replace(["yellow","red"],[1,-1])
    
    return colors_real

def modelTestingCM(colors_real, test_predict, matrix_name):
    #confusion matrix
    colors_real_list = list(colors_real['Color'])
    cm = confusion_matrix(colors_real_list, test_predict)
    cm_matrix = pd.DataFrame(data=cm, columns=['anomaly', 'normal'], 
                                     index=['anomaly', 'normal'])

    
    return cm_matrix
    
    
def getAnomalyScore(data, outlier_thresh, filename):
    
    #scoring for data
    svmModel = SVMModel()
    svm = svmModel.getModel(filename)
    svm.fit_predict(data)
    scores = svm.score_samples(data)
    thresh = quantile(scores, outlier_thresh)
    print(thresh)
    index = where(scores<=thresh)
    values = data['Scaled_Freq'].iloc[index]
    
    colors_frame = pd.DataFrame()
    for i in range(len(data)):
        if(i in values.index):
            temp = pd.DataFrame({"Event": data['Event'][i], "Color": ["red"]})
            colors_frame = colors_frame.append(temp)
        else:
            temp = pd.DataFrame({"Event": data['Event'][i], "Color": ["yellow"]})
            colors_frame = colors_frame.append(temp)
        
    #plotting 
    plt.scatter(data['Event'], data['Scaled_Freq'])
    plt.scatter(values.index, values, color='r')
    plt.show()
    
    
    return colors_frame


freq_table, data, vertices, edges = dataprep()
filename = "svm_model.joblib"
predicted_result = svmFitPredict(data, filename)
balanced_accuracy,weighted_precision,weighted_recall,f_score  = crossValidate(data, predicted_result, filename)
real_colors_global = getBaselineGlobalAnomalies(freq_table)
real_colors_local = getBaselineLocalAnomalies(vertices, edges)

#tested against global thresholds
matrix_name_g = 'Confusion Matrix resulted with global threshold baseline'
svm_matrix_g = modelTestingCM(real_colors_global, predicted_result, matrix_name_g)

#tested against local thersholds
matrix_name_l = 'Confusion Matrix resulted with local threshold baseline'
svm_matrix_l = modelTestingCM(real_colors_local, predicted_result, matrix_name_l)

#get colors frame from anomaly score
colors_frame = getAnomalyScore(data, 0.05, filename)

colors_frame.to_csv("svm_result.csv")







#ROC
# =============================================================================
# X_train, X_test = train_test_split(file,test_size = 0.5,random_state = 42)
# y_score = svm.fit(X_train).decision_function(X_test)
# pred = svm.predict(X_train)
# fpr,tpr,thresholds = roc_curve(pred,y_score)   
# 
# plt.figure(figsize=(6,4))
# plt.plot(fpr, tpr, linewidth=2)
# plt.plot([0,1], [0,1], 'k--' )
# plt.rcParams['font.size'] = 12
# plt.title('ROC curve for one-class SVM')
# plt.xlabel('False Positive Rate (1 - Specificity)')
# plt.ylabel('True Positive Rate (Sensitivity)')
# plt.show()
# =============================================================================
   
