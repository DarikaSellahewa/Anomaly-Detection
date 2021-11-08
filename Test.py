# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 16:52:14 2021

@author: darik
"""
import igraph as ig
import csv
import pandas as pd
import numpy as np
from igraph import Graph
import subgraphMining as subgraph
import matplotlib.pyplot as plt
import numpy as np
import FrequencyCalc as frequencyCalc
import ColorAssign as colorAssign
from datetime import datetime, timedelta
import preProcess as preprocess
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
import NaiveBayes as naiveBayes
import preProcess as preProcess
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)


file = pd.read_csv("Data file.csv", delimiter=",")
#file = file[0:200]
log_event = preProcess.preProcess(file)
vertices = subgraph.getVertices(log_event)
edges = subgraph.getEdges(vertices)
freqTable = frequencyCalc.getGlobalFreqTable(vertices, edges)
high,medium = frequencyCalc.getGlobalThresholds(freqTable)
color_frame = colorAssign.getGlobalColours(freqTable, high, medium)
filename = "Completed_model.joblib"
model = naiveBayes.modelTrain(freqTable, color_frame, filename)

class_table = freqTable.sort_values(by = ['Event'])

#replace index column
class_table.reset_index(drop=True, inplace=True)
events = freqTable.index

#adding event column
class_table[['Event']] = list(events.sort_values(ascending = True))


model.predict()


#testing for nighbourgraph vis
eventFreqTable = frequencyCalc.getLocalFreqTable(vertices, edges)
eventFreqTable.reset_index(drop = True, inplace = True)
temp = eventFreqTable[eventFreqTable.Event == 8002]
temp = temp.sort_values(by = ["Frequency"])
ind = temp.tail(1).index.item()




#test NB
data = fetch_20newsgroups()
categories = ['talk.religion.misc', 'soc.religion.christian',
              'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)

test = fetch_20newsgroups(subset='test', categories=categories)

test_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
test_model.fit(train.data, train.target)
test_labels = test_model.predict(test.data)

# =============================================================================
# #sampling
# train_data = class_table.sample(frac = .75)
# test_data = class_table.sample(frac = .25)
# 
# =============================================================================

#testing dic vectoriser
df = pd.DataFrame({'col1': [1, 2],
                   'col2': [0.5, 0.75]})
d = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
c = df.to_dict()
v = DictVectorizer(sparse=False)
X = v.fit_transform(df.to_dict('records'))

data = [
{'house': 100, 'street': 50, 'shop': 25, 'car': 100, 'tree': 20},

{'house': 5, 'street': 5, 'shop': 0, 'car': 10, 'tree': 500, 'river': 1}
]
Y = np.array([1, 0]) 


dv = DictVectorizer()
mnb = MultinomialNB()
mnb.fit(dv.fit_transform(data),Y)
mnb.predict(dv.fit_transform(data))
#validation
#testing how to keep model accessibility
test_model = make_pipeline(DictVectorizer() , MultinomialNB())
test_model.fit(data,Y )
test_model.predict(data)




#Test
# Create graph
g_test = Graph(directed = True)
vert_test = ["1","2","3","4","5"]
g_test.vs["label"] = vert_test
g_test.add_vertices(5)
edges_test = []
edges_test = [(0,3),(1,2),(2,4),(3,4),(4,4),(4,1)]
g_test.add_edges(edges_test)
g_test.es["trans"] = edges_test
print(g_test)
ig.plot(g_test)

#subgraph

#Frequency
g_test.incident(3)
g_test.neighborhood(3)
g_test.successors(0)
g_test.predecessors(10)
g_test.subgraph(1)
ig.plot(g_test.subgraph([4,5]))
g_test.vertex_connectivity(target =2, source =1)


#hist
plt.hist(frequencyCalc.fre_table["count"],density=True)


#date time
appointments = [(datetime(2012, 5, 22, 10), datetime(2012, 5, 22, 10, 30)),
                (datetime(2012, 5, 22, 12), datetime(2012, 5, 22, 13)),
                (datetime(2012, 5, 22, 15, 30), datetime(2012, 5, 22, 17, 10))]
if datetime(2012, 5, 22, 10) > datetime(2012, 5, 22, 10, 30):
    print("Yes")

test_data = preprocess.log_unsorted[preprocess.log_unsorted[' DATE'] < datetime(2021, 9, 2, 8)]
# =============================================================================
# csvreader = csv.reader(file)
# header = []
# header = next(csvreader)
# header
# =============================================================================


#changing the column type to string
# =============================================================================
# class_table = class_table.astype({"Event" : str})
# =============================================================================

#mapping int to colors
# =============================================================================
# mapping = {'red' : 1, 'green' : 2, 'yellow' : 3}
# class_table = class_table.replace({'Color': mapping})
# =============================================================================

