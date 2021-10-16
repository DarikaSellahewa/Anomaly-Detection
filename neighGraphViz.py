# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 22:31:08 2021

@author: darik
"""

import GraphViz as graphViz
from igraph import Graph




def getNeighbourGraph(event, levels, eventFreqTable):
    
    eventFreqTable.reset_index(drop = True, inplace = True)
    temp = eventFreqTable[eventFreqTable.Event == event]
    temp = temp.sort_values(by = ["Frequency"])
    ind = temp.tail(1).index.item()
    
    #creating graph
    nei_graph = Graph(directed=True)
    vertices = eventFreqTable["Event"]
    nei_graph.add_vertices(len(vertices)) 
    for i in range(len(nei_graph.vs)):
        nei_graph.vs[i]["id"]= i
        nei_graph.vs[i]["label"]= str(vertices.iloc[i]) 
    

    edges = []
    for i in range(len(vertices)-1):
        edges.append((i,i+1))
       
    nei_graph.add_edges(edges)
    
    #Neighbors of high frequency events
    neighbors = nei_graph.neighborhood(ind,levels)
        
        
    visual_style = {}
    
    out_name = "neighbor_graph of "+ str(event) + ".png"
    # Set bbox and margin
    visual_style["bbox"] = (1200,700)
    visual_style["margin"] = 27
    # Set vertex size
    visual_style["vertex_size"] = 30
    # Set vertex lable size
    visual_style["vertex_label_size"] = 10
    
    # Set vertex colours
    visual_style["vertex_color"] = "blue"
    
    # Set the layout
    my_layout = nei_graph.layout_lgl()
    visual_style["layout"] = my_layout
    
    graphViz.ig.plot(nei_graph.subgraph(neighbors),out_name,**visual_style)
    
    

# =============================================================================
# #add edges
# gr.add_edges(edges)
# 
# =============================================================================


# =============================================================================
# #Neighbors of medium frequency events
# for u in med_freq:
#     neighbors = subgraphMining.gg.neighborhood(u,levels)
#  
# 
# =============================================================================
