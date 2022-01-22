# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 16:30:56 2021

@author: darik
"""

import pandas as pd

def preProcess(logFile):

    #file = pd.read_csv("Search Results.csv", delimiter=",")
    file = logFile
    #filtering teh meaningful data
    log_data = file.loc[:,[' DATE',' TIME',' SYSTEM',' TABLE',' DATA',
                       ' EVENTCOUNT',' EVENTID',' RETURN',' SNAREDATAMAP',
                       ' SOURCE',' SOURCETYPE',' STRINGS',' USER',' ACTION',' DETAILS',
                       ' RESOURCE',' USERNAME' ]]

    #filtering data from the host machine and snare server
    log_event = file[file[" SYSTEM"].isin(["HOST_NAME","IP_SNARE_SERVER"])]

    #filtering data columns
    log_event = file.loc[:,[" SYSTEM", " DATE", " TIME",  " EVENTID"]] 
    #replace '-' with 0 where snare logs miss events id
    log_event[" EVENTID"].replace("-", 0, inplace = True)
    log_event[" EVENTID"].replace("ADMINISTRATOR", 0, inplace = True)

    #remove nan values
    log_event.dropna(subset=[' EVENTID'], inplace = True)
    
    #sort with date and time
    log_event['TIMESTAMP'] = pd.to_datetime(log_event[' DATE'] + ' ' + log_event[' TIME'],dayfirst= True)
    log_event.sort_values(by = ['TIMESTAMP'])
    
    #seleting the time frame data
    log_event.index = pd.to_datetime(log_event[' DATE'] + ' ' + log_event[' TIME'])
    log_event = log_event.between_time ('8:00', '20:00').reset_index(drop=True)

    return log_event

