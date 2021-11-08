# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:29:31 2021

@author: darik
"""
from igraph import Graph
import SVM as svm
import igraph as ig


#visualisation
unique_vert = svm.colors_frame['Event']
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
    anomaly_graph.vs[i]["label"]= str(list(unique_vert)[i]) 


visual_style = {}
out_name = "graph_svm.png"
# Set bbox and margin
visual_style["bbox"] = (1200,700)
visual_style["margin"] = 27
# Set vertex size
visual_style["vertex_size"] = 30
# Set vertex lable size

visual_style["vertex_label_size"] = 10
#anomaly_graph.vs["color"] = colors
# Set vertex colours
visual_style["vertex_color"] = svm.colors_frame['Color']

# Set the layout
my_layout = anomaly_graph.layout_lgl()
visual_style["layout"] = my_layout

#add edges
edges = []
for i in range(len(unique_vert)-1):
    edges.append((i,i+1))
anomaly_graph.add_edges(edges)
#anomaly.ig.plot(anomaly_graph, outname, **visual_style)
ig.plot(anomaly_graph,out_name, **visual_style)

    
    

