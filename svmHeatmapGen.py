# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 12:13:36 2021

@author: darik
"""
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
import SVM as svm
import dill

svm_matrix_g = svm.svm_matrix_g
svm_matrix_g = pd.DataFrame(data=svm_matrix_g, columns=['anomaly', 'normal'], 
                                 index=['anomaly', 'normal'])

sns.heatmap(svm_matrix_g, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix resulted with global threshold baseline')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()

total_events = svm_matrix_g['anomaly']['anomaly'] + svm_matrix_g['normal']['anomaly'] + svm_matrix_g['anomaly']['normal'] + svm_matrix_g['normal']['normal']
svm_sensitivity_g =  svm_matrix_g['anomaly']['anomaly']/(svm_matrix_g['anomaly']['anomaly'] + svm_matrix_g['normal']['anomaly'])
svm_specificity_g = svm_matrix_g['normal']['normal']/(svm_matrix_g['normal']['normal'] + svm_matrix_g['anomaly']['normal'])
svm_balanced_acc_g = (svm_sensitivity_g + svm_specificity_g)/2
svm_precision_g = svm_matrix_g['anomaly']['anomaly']/(svm_matrix_g['anomaly']['anomaly'] + svm_matrix_g['anomaly']['normal'])
svm_type_I_error_g = svm_matrix_g['anomaly']['normal']/total_events
svm_type_II_error_g = svm_matrix_g['normal']['anomaly']/total_events


#local
svm_matrix_l = svm.svm_matrix_l
svm_matrix_l = pd.DataFrame(data=svm_matrix_l, columns=['anomaly', 'normal'], 
                                 index=['anomaly', 'normal'])

sns.heatmap(svm_matrix_l, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix resulted with local threshold baseline')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


svm_sensitivity_l =  svm_matrix_l['anomaly']['anomaly']/(svm_matrix_l['anomaly']['anomaly'] + svm_matrix_l['normal']['anomaly'])
svm_specificity_l = svm_matrix_l['normal']['normal']/(svm_matrix_l['normal']['normal'] + svm_matrix_l['anomaly']['normal'])
svm_balanced_acc_l = (svm_sensitivity_l + svm_specificity_l)/2
svm_precision_l = svm_matrix_l['anomaly']['anomaly']/(svm_matrix_l['anomaly']['anomaly'] + svm_matrix_l['anomaly']['normal'])
svm_type_I_error_l = svm_matrix_l['anomaly']['normal']/total_events
svm_type_II_error_l = svm_matrix_l['normal']['anomaly']/total_events

filename = 'globalsave_svm.pkl'
dill.dump_session(filename)