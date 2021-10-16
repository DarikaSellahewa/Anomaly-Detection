# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 16:25:53 2021

@author: darik
"""

import igraph as ig
import pandas as pd
import numpy as np
from igraph import Graph
import preProcess as preProcess


def getVertices(eventFrame):
    
    #setting vertices values
    #convert eventid to integers
    vertices = eventFrame[" EVENTID"].apply(pd.to_numeric).astype(int)
    
    return vertices
    
def getEdges(vertices):
    
    #setting edges values
    edges = []
    for i in range(len(vertices)-1):
        edges.append((i,i+1))
    
    return edges

# =============================================================================
# # Create graph
# gg = Graph(directed = True)
# 
# #convert eventid to integers
# subGraphEvents = preProcess.log_event[" EVENTID"].apply(pd.to_numeric).astype(int)
# 
# #setting vertices values
# vertices = subGraphEvents
# 
# #setting edges values
# edges = []
# for i in range(len(vertices)-1):
#     edges.append((i,i+1))
# 
# gg.add_vertices(vertices)
# gg.add_edges(edges)
# print(gg)
# ig.plot(gg)
# 
# 
# =============================================================================

