# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 16:02:12 2021

@author: darik
"""

import pandas as pd
import igraph as ig
from igraph import Graph
import FrequencyCalc as frequencyCalc
import subgraphMining as subgraphMining
import preProcess as preProcess
import NaiveBayes as naiveBayes


    
# =============================================================================
# #reading dataset as batch file
# file = pd.read_csv("Search Results.csv", delimiter=",")
# log_event = preProcess.preProcess(file)
# 
# unique_vert = pd.Series(subgraphMining.getVertices(log_event).unique())
# global_color_frame = pd.DataFrame()
# event_color_frame = pd.DataFrame()
#     
# 
# =============================================================================
#Anomaly detection

def globalVisualisation(vertices, filename):
        
        unique_vert = vertices.unique()
        #setting labels into vertices
        labels = []
        for x in unique_vert:
            labels.append(str(x))
            pass
        
        edges = []
        for i in range(len(vertices)-1):
            edges.append((i,i+1))
        
        
        anomaly_graph = Graph(directed = True)
        anomaly_graph.add_vertices(len(unique_vert)) 
        for i in range(len(anomaly_graph.vs)):
            anomaly_graph.vs[i]["id"]= i
            anomaly_graph.vs[i]["label"]= str(unique_vert[i]) 
        
        fre_table = frequencyCalc.getGlobalFreqTable(vertices,edges)
        colors = naiveBayes.getPrediction(fre_table, filename)
        
        visual_style = {}
        out_name = "graph.png"
        # Set bbox and margin
        visual_style["bbox"] = (1200,700)
        visual_style["margin"] = 27
        # Set vertex size
        visual_style["vertex_size"] = 30
        # Set vertex lable size
        
        visual_style["vertex_label_size"] = 10
        #anomaly_graph.vs["color"] = colors
        # Set vertex colours
        visual_style["vertex_color"] = colors
        
        # Set the layout
        my_layout = anomaly_graph.layout_lgl()
        visual_style["layout"] = my_layout
        
        #add edges
        edges = []
        for i in range(len(unique_vert)-1):
            edges.append((i,i+1))
        anomaly_graph.add_edges(edges)
        #anomaly.ig.plot(anomaly_graph, outname, **visual_style)
        ig.plot(anomaly_graph, **visual_style)
    
        return anomaly_graph, visual_style, fre_table, colors

    
def graphlVisualisation(freqTable, colorFrame):
        
        unique_vert = freqTable.index
        #setting labels into vertices
        labels = []
        for x in unique_vert:
            labels.append(str(x))
            pass
        
        edges = []
        for i in range(len(unique_vert)-1):
            edges.append((i,i+1))
        
        
        anomaly_graph = Graph(directed = True)
        anomaly_graph.add_vertices(len(unique_vert)) 
        for i in range(len(anomaly_graph.vs)):
            anomaly_graph.vs[i]["id"]= i
            anomaly_graph.vs[i]["label"]= str(unique_vert[i]) 
        

        visual_style = {}
        out_name = "graph.png"
        # Set bbox and margin
        visual_style["bbox"] = (1200,700)
        visual_style["margin"] = 27
        # Set vertex size
        visual_style["vertex_size"] = 30
        # Set vertex lable size
        
        visual_style["vertex_label_size"] = 10
        #anomaly_graph.vs["color"] = colors
        # Set vertex colours
        visual_style["vertex_color"] = colorFrame
        
        # Set the layout
        my_layout = anomaly_graph.layout_lgl()
        visual_style["layout"] = my_layout
        
        #add edges
        edges = []
        for i in range(len(unique_vert)-1):
            edges.append((i,i+1))
        anomaly_graph.add_edges(edges)
        #anomaly.ig.plot(anomaly_graph, outname, **visual_style)
        ig.plot(anomaly_graph, **visual_style)
    
        return anomaly_graph, visual_style







  
















