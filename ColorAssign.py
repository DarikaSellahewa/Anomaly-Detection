# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 20:59:01 2021

@author: darik
"""

import pandas as pd
import FrequencyCalc as frequencyCalc


def getGlobalColours(fre_table, high_threshold,medium_threshold):
    #using global threshold
    global_color_frame = pd.DataFrame()
    unique_vert = fre_table.index
    for j in unique_vert:
        if fre_table["Freq_dist"][j] > high_threshold:
            temp = pd.DataFrame({"Event": [j], "Color": ["red"]})
            global_color_frame = global_color_frame.append(temp)
        elif fre_table["Freq_dist"][j] > medium_threshold:
            temp = pd.DataFrame({"Event": [j], "Color": ["green"]})
            global_color_frame = global_color_frame.append(temp)
        else:
            temp = pd.DataFrame({"Event": [j], "Color": ["yellow"]})
            global_color_frame = global_color_frame.append(temp)
        pass
    
    global_color_frame = global_color_frame.set_index('Event')
    
    return global_color_frame


def getLocalColors(fre_table, threshold_table):
    event_color_frame = pd.DataFrame()
    # Global frequency table
    unique_vert = fre_table.index
    for j in unique_vert:
        high_event_thresh = threshold_table[threshold_table.Event == j]['High Threshold'][0]
        medium_event_thresh = threshold_table[threshold_table.Event == j]['Medium Threshold'][0]
    
        if high_event_thresh > 1 and fre_table["Frequency"][j] > high_event_thresh:
            temp = pd.DataFrame({"Event": [j], "Color": ["red"]})
            event_color_frame = event_color_frame.append(temp)
        elif medium_event_thresh > 1 and fre_table["Frequency"][j] > medium_event_thresh:
            temp = pd.DataFrame({"Event": [j], "Color": ["green"]})
            event_color_frame = event_color_frame.append(temp)
        else:
            temp = pd.DataFrame({"Event": [j], "Color": ["yellow"]})
            event_color_frame = event_color_frame.append(temp)
        pass

    return event_color_frame

  
















