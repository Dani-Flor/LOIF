import pandas as pd
from sklearn.metrics import classification_report  #, f1_score,recall_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import ast
import time
import tkinter as tk
from tkinter import filedialog
import re
import random

# #This function create a KNN model where we specify the which features we are interested in and the outages we want to test using k neighbors
def lod_exp_exec(Training_all,otl_col,outage_labels,k,output,f):
    #In training data we filter out data to only look at the obsevation points (Features) and outages we are interested in.
    Training_BD = Training_all.loc[outage_labels,otl_col]

    # Here we use MinMaxscaler to convert our measurments into values from 0 to 1
    X = Training_BD.iloc[:,:-1]  ## Features
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
    # print(X)
    y = Training_BD.iloc[:,-1]  ## Labels 

    #From the same training data, we split it into training and testing data to run KNN
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

    #Set KNN classifier, fit our model and make predictions on testing labels
    knn = KNeighborsClassifier(n_neighbors=k,weights='uniform',algorithm='kd_tree',p=2,)  #p=2 is euclidean distance, p=1 is manhattan distance
    #Train KNN with training data from split
    knn.fit(x_train, y_train)
            #X_train, y_train
    Predictions = knn.predict(x_test)
    True_Labels = y_test

    #Collect classification report between predicted labels and real labels from testing data
    Report= classification_report(True_Labels,Predictions,output_dict=True,zero_division=0)
    if output == True:
        print(classification_report(True_Labels,Predictions),file=f)
    return Report

def max_coverage_greedy(mgb_df,OTL_num,f,Training_all):
    print('Starting MCP Algorithm......')
    OTLs = []
    # print(mgb_df)
    print('-----------------MCP GREEDY ALGORITHM-------------------',file=f)
    collection_set = []  #Will be used to store the outages we have already covered
    for i in range(0,OTL_num):
        diffs = {} #Will hold the number of new elements each subset has so that we can compare all of them at the end
        

        #This for loop will go through Sa subsets to get the difference between each subset and and the collection set for current iteration
        for j in mgb_df.index:  #For every observation point
            current_set = mgb_df.loc[j,'Sa_Subsets']
            current_set = ast.literal_eval(current_set)

            #Calculate the diffrence between current iteration sets and collection set to see which outages are not covered yet
            diffs[f'{j}'] = len(list(set(current_set).difference(collection_set)))   #Counts the number of uncovered outages

        largest_OPL = max(diffs, key=diffs.get)   #Gets the Sa subset with the largest number of elements not yet covered in collection set.
        print(f'Largest Subset (Iteration {i+1}): {largest_OPL} --> {diffs[largest_OPL]}',file=f)
        if diffs[largest_OPL] == 0:  #If there are no more new elements stop iteration
            break
        else:                        #If there are still uncovered elements continue iteration and update collection set.
            OTLs.append(int(largest_OPL))
            current_set = mgb_df.loc[int(largest_OPL),'Sa_Subsets']
            current_set = ast.literal_eval(current_set)
            collection_set = list(set(collection_set + current_set))

    print(f'OTLs ({len(OTLs)}):\n',OTLs,file=f)

    print(f'Covered Outages({len(collection_set)}):\n',collection_set,file=f)
    otl_col = []
    for i in OTLs:
        otl_col.append(f'PF Line {i}')  
        otl_col.append(f'QF Line {i}')
        otl_col.append(f'PT Line {i}')
        otl_col.append(f'QT Line {i}')
    otl_col.append(f'Label')

    outage_labels = [0] + collection_set

    print(mgb_df.loc[OTLs,:],file=f)
    Report = lod_exp_exec(k=8,otl_col=otl_col,outage_labels=outage_labels,output=True,f=f,Training_all=Training_all)
    print('---------------------End Of MCP GREEDY ALGORITHM----------------------',file=f)
    return [OTLs,collection_set,Report['macro avg']['f1-score']]

def high_eta(mgb_df,MCP,f,Training_all):
    print('Starting High Eta Tests.......')
    print('------------------HIGH ETA Test-----------------------------------------',file=f)
    # print(mgb_df)
    # print(MCP)
    Eta_OTLs = []
    for i in mgb_df.index[:len(MCP[0])]:
        Eta_OTLs.append(i)
    print(f'High Eta Lines ({len(Eta_OTLs)}): {Eta_OTLs}',file=f)
    print(f'Outage Set, Same as MCP results ({len(MCP[1])}): {MCP[1]}',file=f)
    otl_col = []
    for j in Eta_OTLs:
        otl_col.append(f'PF Line {j}')  
        otl_col.append(f'QF Line {j}')
        otl_col.append(f'PT Line {j}')
        otl_col.append(f'QT Line {j}')
    otl_col.append(f'Label')

    outage_labels = [0] + MCP[1]
    lod_exp_exec(k=8,otl_col=otl_col,outage_labels=outage_labels,output=True,f=f,Training_all=Training_all)
    print('-----------------End of HIGH ETA Test---------------------------------',file=f)

def random_OTL(MCP,f,Training_all,LOIF):
    print('Starting Random OTL Tests......')
    print('------------------RANDOM OTL SELECTION------------------------------------',file=f)
    scores = []
    for i in range(1,11):
        print('\n',file=f)
        print(f'Random Test {i}',file=f)
        rand_OTLs = random.sample(LOIF.index.to_list(), len(MCP[0]),)
        print(f'Random OTLs ({len(rand_OTLs)}): {rand_OTLs}',file=f)
        print(f'Outage Set, Same as MCP results ({len(MCP[1])}): {MCP[1]}',file=f)

        otl_col = []
        for j in rand_OTLs:
            otl_col.append(f'PF Line {j}')  
            otl_col.append(f'QF Line {j}')
            otl_col.append(f'PT Line {j}')
            otl_col.append(f'QT Line {j}')
        otl_col.append(f'Label')
        outage_labels = [0] + MCP[1]
        Report = lod_exp_exec(k=8,otl_col=otl_col,outage_labels=outage_labels,output=False,f=f,Training_all=Training_all)
        print(f'Overall F1-Score (Test {i}): {Report['macro avg']['f1-score']}',file=f)
        scores.append(Report['macro avg']['f1-score'])
    print('\n',file=f)
    avg_score = sum(scores)/len(scores)

    print(f'Average F1-Score ({i} Tests): {avg_score}',file=f)
    print('-------------------END of RANDOM OTL SELECTION---------------------------',file=f)

def lod_otl_select(mgb_df,f,Training_all,LOIF,OTL_num):
    if OTL_num == "FC":
        OTL_num = 3000

    Results = {}
    print(f'Final Sa Subsets:\n {mgb_df}',file=f)
    print('\n',file=f)
    Results['MCP'] = max_coverage_greedy(mgb_df=mgb_df.sort_values(by='Total_outages',ascending=False),OTL_num=OTL_num,f=f,Training_all=Training_all)
    print('\n',file=f)
    Results['Eta'] = high_eta(mgb_df=mgb_df.sort_values(by='Total_outages',ascending=False),MCP= Results['MCP'],f=f,Training_all=Training_all)
    print('\n',file=f)
    Results['Random'] = random_OTL(MCP=Results['MCP'],f=f,Training_all=Training_all,LOIF=LOIF)
