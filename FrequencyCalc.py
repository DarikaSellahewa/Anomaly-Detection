# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 11:24:33 2021

@author: darik
"""

import pandas as pd
from sklearn import preprocessing
from igraph import Graph
import subgraphMining as subgraph

def getGlobalFreqTable(vertices, edges):
    gg = Graph(directed = True)
    gg.add_vertices(vertices)
    gg.add_edges(edges)
    frequency_frame = pd.DataFrame()
    for edge in range(len(edges)):
        if(edge == 0):
            frequency_frame = pd.DataFrame(
                {"Event": [edges[0][0]], "Frequency": [1]})

        edg = gg.es[edge]
        v_1 = vertices[edg.source]
        v_2 = vertices[edg.target]
        temp_df = pd.DataFrame({"Event": [v_2], "Frequency": [1]})
        frequency_frame = frequency_frame.append(temp_df)

    # Global frequency table
    fre_table = pd.crosstab(index=frequency_frame['Event'], columns='Frequency')
    # calculating freqency distribution
    fre_table[['Freq_dist']] = fre_table['Frequency']/sum(fre_table['Frequency'])

    # noramlizing data
    scaler = preprocessing.MinMaxScaler()
    fre_table[['Scaled_Freq']] = scaler.fit_transform(fre_table[['Frequency']])


    return fre_table

def getLocalFreqTable(vertices, edges):
    gg = Graph(directed = True)
    gg.add_vertices(vertices)
    gg.add_edges(edges)
    local_fre_frame = pd.DataFrame()
    temp_fre = 1
    for edge in range(len(edges)):
        edg = gg.es[edge]
        v_1 = vertices[edg.source]
        v_2 = vertices[edg.target]
        
        if v_1 == v_2:
            temp_fre += 1
        else:
            temp_fre_dist = pd.DataFrame(
                {"Event": [v_1], "Frequency": [temp_fre]})
            local_fre_frame = local_fre_frame.append(temp_fre_dist)
            temp_fre = 1 
            
    return local_fre_frame

def getGlobalThresholds(frequencyTable):
    # calculating global frequency threshold
    fre_table = frequencyTable.sort_values(by=["Frequency"])


    high_threshold = fre_table["Scaled_Freq"].quantile(.97, interpolation="midpoint")
    medium_threshold = fre_table["Scaled_Freq"].quantile(.20, interpolation="midpoint")
    
    return high_threshold, medium_threshold

def freqTableAdjust(fre_Table):
    # Global frequency table
    fre_table = pd.crosstab(index=fre_Table['Event'], columns='Frequency')
    # calculating freqency distribution
    fre_table[['Freq_dist']] = fre_table['Frequency']/sum(fre_table['Frequency'])

    # noramlizing data
    scaler = preprocessing.MinMaxScaler()
    fre_table[['Scaled_Freq']] = scaler.fit_transform(fre_table[['Frequency']])

    return fre_table

def getLocalThresholds(localFreqTable):
    event_threshold_table = pd.DataFrame()
    adjFreqTable = pd.DataFrame()
    localFreqTable = localFreqTable.sort_values(by=["Event"])
    for event in localFreqTable['Event'].unique():
        temp_event = localFreqTable[localFreqTable.Event == event]
        
        # calculating freqency distribution
        temp_event[['Freq_dist']] = temp_event['Frequency']/sum(temp_event['Frequency'])
    
    
        # normalising data
        event_scaler = preprocessing.MinMaxScaler()
        temp_event[['Scaled_Freq']] = event_scaler.fit_transform(temp_event[['Frequency']])
    
        high_event_threshold = temp_event.sort_values(
            by=['Frequency']).quantile(.99, interpolation='midpoint')
        medium_event_threshold = temp_event.sort_values(
            by=['Frequency']).quantile(.05, interpolation='midpoint')
        temp_thresh = pd.DataFrame({"Event": [event], "High Threshold": [
                                   high_event_threshold['Frequency']], "Medium Threshold": [medium_event_threshold['Frequency']]})
        event_threshold_table = event_threshold_table.append(temp_thresh)
        #temp_frq_event = pd.DataFrame({"Event": [event], "Freq_dist": [temp_event['Freq_dist'][0]], "Scaled_freq": [temp_event['Scaled_freq'][0]]})
        #adjFreqTable = adjFreqTable.append(temp_frq_event)    
    adjFreqTable = freqTableAdjust(localFreqTable)
    
    #adjFreqTable.set_index('Event')
    return event_threshold_table, adjFreqTable

