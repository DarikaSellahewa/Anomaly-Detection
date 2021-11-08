# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 09:38:04 2021

@author: darik
"""
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
import dill                            #pip install dill --user


#creating confusion matrix heat map

#global
cm_g = [[7,0],[81,123]]
cm_matrix_g = pd.DataFrame(data=cm_g, columns=['anomaly', 'normal'], 
                                 index=['anomaly', 'normal'])

sns.heatmap(cm_matrix_g, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix resulted with global threshold baseline')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()

total_events = cm_matrix_g['anomaly']['anomaly']+cm_matrix_g['normal']['anomaly']+cm_matrix_g['anomaly']['normal']+cm_matrix_g['normal']['normal']
sensitivity_g =  cm_matrix_g['anomaly']['anomaly']/(cm_matrix_g['anomaly']['anomaly'] + cm_matrix_g['normal']['anomaly'])
specificity_g = cm_matrix_g['normal']['normal']/(cm_matrix_g['normal']['normal'] + cm_matrix_g['anomaly']['normal'])
balanced_acc_g = (sensitivity_g + specificity_g)/2
precision_g = cm_matrix_g['anomaly']['anomaly']/(cm_matrix_g['anomaly']['anomaly'] + cm_matrix_g['anomaly']['normal'])
type_I_error_g = cm_matrix_g['anomaly']['normal']/total_events
type_II_error_g = cm_matrix_g['normal']['anomaly']/total_events


#local
cm_l = [[17,54],[6,134]]
cm_matrix_l = pd.DataFrame(data=cm_l, columns=['anomaly', 'normal'], 
                                 index=['anomaly', 'normal'])

sns.heatmap(cm_matrix_l, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix resulted with local threshold baseline')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


sensitivity_l =  cm_matrix_l['anomaly']['anomaly']/(cm_matrix_l['anomaly']['anomaly'] + cm_matrix_l['normal']['anomaly'])
specificity_l = cm_matrix_l['normal']['normal']/(cm_matrix_l['normal']['normal'] + cm_matrix_l['anomaly']['normal'])
balanced_acc_l = (sensitivity_l + specificity_l)/2
precision_l = cm_matrix_l['anomaly']['anomaly']/(cm_matrix_l['anomaly']['anomaly'] + cm_matrix_l['anomaly']['normal'])
type_I_error_l = cm_matrix_l['anomaly']['normal']/total_events
type_II_error_l = cm_matrix_l['normal']['anomaly']/total_events

filename = 'globalsave.pkl'
dill.dump_session(filename)



