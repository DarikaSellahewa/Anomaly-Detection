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


#reading dataset as batch file
file = pd.read_csv("Data file.csv", delimiter=",")
file = file.sample(frac = .75)
log_event = preProcess.preProcess(file)
vertices = subgraph.getVertices(log_event)
filename = "Completed_model_l.joblib"
anomaly_graph, visual = graphViz.globalVisualisation(vertices,filename)
ig.plot(anomaly_graph, target= 'graph_result.pdf', **visual)
