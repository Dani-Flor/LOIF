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
####################################### KNN classifier #########################################################
# This function creates a KNN model where we specify which features ("otl_col") we are interested in and the outages ("outage_labels") 
# we want to test using "k" neighbors. This function is used in "lod_detectable_subsetgen" where the outages selected depend on gamma and beta,
# and "lod_otl_select" where we test all possible outages. The KNN results are saved in output text file "f" if "output" is true (print results, do not if false).
# We use "training_data" to predict outage labels, but first split the data to obtain testing data using "train_test_split".
# 
# Parameters
# - otl_col: The columns in training data representing the OTLs that are selected in "lod_detectable_subsetgen" and "lod_otl_select"
# - outage_labels: The rows in training data representing the outages we are interested in. Outages from "lod_detectable_subsetgen" depend on gamma and beta.
#                  We consider all outages in "lod_otl_select".
# - k: Number of Neighbors to consider in KNN
# - f: this is the text file where we save our KNN results
# - output: TRUE/FALSE, if true save results in text file "f"
# - training_data: This is the labeled power flow data obtained from MATLAB script "lod_labeled_datagen.m"
def lod_exp_exec(otl_col,outage_labels,k,f,output,training_data):
    #In training data we filter out data to only look at the obsevation points (Features) and outages we are interested in.
    Training_BD = training_data.loc[outage_labels,otl_col]

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

########################## MCP OTL Selection #####################################
# Using the results we obtained from "lod_detectable_subsetgen" (in "mgb_df"). We perform an iterative process where we select subsets
# that offer the most uncoverged outages in each iteration. These outages are stored in a collection set (initially empty). This process is repeated
# until we have selected a specific number of OTLs (based on "OTL_num") or until we have covered all possible outages (Full Coverage).
# Once the OTLs are determined, we use our "training_data" to run KNN classifier with "k" neighbors for all possible outages (in "outage_set"), and save
# KNN results in output text file "f".
#
# Parameters
# - mgb_df: This dataframe holds the results from function "lod_detectable_subsetgen"
# - OTL_num: Number of OTLs to select (From main [1,2,4,8,Full Coverage])
# - training_data: This is the labeled power flow data obtained from MATLAB script "lod_labeled_datagen.m"
# - k: Number of Neighbors to consider in KNN
# - outage_set: Set of all outages that converged
# - f: this is the text file where we save our KNN results

def mcp(mgb_df,OTL_num,training_data,k,outage_set,f):
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
        if diffs[largest_OPL] == 0:  #If there are no more uncovered outages stop iteration
            break
        else:                        #If there are still uncovered outages, continue iteration and update collection set.
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

    outage_labels = [0] + outage_set

    print(mgb_df.loc[OTLs,:],file=f)
    Report = lod_exp_exec(k=k,otl_col=otl_col,outage_labels=outage_labels,output=True,f=f,training_data=training_data)
    print('---------------------End Of MCP GREEDY ALGORITHM----------------------',file=f)
    return [OTLs,collection_set,Report['macro avg']['f1-score']]

########################### High Eta OTL selection ###########################################
# This method selects the same number of OTLs as "MCP" using the results from our detectable subset generation (in "mgb_df") 
# to select the largest subsets. After this we use our "training_data" to run KNN classifier with "k" neighbors for all possible outages in
# "outage_set", and saves results into output text file "f".
#
# Parameters
# - MCP: Results of MCP, used only to select the same number of OTLs. Has to be done this way because Full coverage is based on MCP results.
# - mgb_df: This dataframe holds the results from function "lod_detectable_subsetgen"
# - training_data: This is the labeled power flow data obtained from MATLAB script "lod_labeled_datagen.m"
# - k: Number of Neighbors to consider in KNN
# - outage_set: Set of all outages that converged.
# - f: this is the text file where we save our KNN results
def high_eta(MCP,mgb_df,training_data,k,outage_set,f):
    print('Starting High Eta Tests.......')
    print('------------------HIGH ETA Test-----------------------------------------',file=f)

    Eta_OTLs = []
    for i in mgb_df.index[:len(MCP[0])]:
        Eta_OTLs.append(i)
    print(f'High Eta Lines ({len(Eta_OTLs)}): {Eta_OTLs}',file=f)
    print(f'Outage Set, Same as MCP results ({len(outage_set)}): {outage_set}',file=f)
    otl_col = []
    for j in Eta_OTLs:
        otl_col.append(f'PF Line {j}')  
        otl_col.append(f'QF Line {j}')
        otl_col.append(f'PT Line {j}')
        otl_col.append(f'QT Line {j}')
    otl_col.append(f'Label')

    outage_labels = [0] + outage_set
    Report = lod_exp_exec(k=k,otl_col=otl_col,outage_labels=outage_labels,output=True,f=f,training_data=training_data)
    print('-----------------End of HIGH ETA Test---------------------------------',file=f)
    return Report['macro avg']['f1-score']

############################Random OTL selection##############################
# This method selects the same number of OTLs that were selected in "MCP" (same as OTL_num). Uses "LOIF" to get a set of all
# OTLs then selects a random sample from this set. Then we use "training_data" to run KNN classifier with "k" neighbors, and save results
# in output text file "f". We run KNN for all possible outages (from "outage_set") that converged.
#
# Parameters
# - MCP: Results of MCP, used only to select the same number of OTLs. Has to be done this way because Full coverage is based on MCP results.
# - training_data: This is the labeled power flow data obtained from MATLAB script "lod_labeled_datagen.m"
# - k: Number of Neighbors to consider in KNN
# - f: this is the text file where we save our KNN results
# - LOIF: This the LOIF matrix of the system exlcuding data from outages that don't converge.
# - outage_set: Set of all outages that converged.

def random_otl(MCP,training_data,k,f,LOIF,outage_set):
    print('Starting Random OTL Tests......')
    print('------------------RANDOM OTL SELECTION------------------------------------',file=f)
    scores = []
    for i in range(1,11):
        print('\n',file=f)
        print(f'Random Test {i}',file=f)
        rand_OTLs = random.sample(LOIF.index.to_list(), len(MCP[0]),)
        print(f'Random OTLs ({len(rand_OTLs)}): {rand_OTLs}',file=f)
        print(f'Outage Set, Same as MCP results ({len(outage_set)}): {outage_set}',file=f)

        otl_col = []
        for j in rand_OTLs:
            otl_col.append(f'PF Line {j}')  
            otl_col.append(f'QF Line {j}')
            otl_col.append(f'PT Line {j}')
            otl_col.append(f'QT Line {j}')
        otl_col.append(f'Label')
        outage_labels = [0] + outage_set
        Report = lod_exp_exec(k=k,otl_col=otl_col,outage_labels=outage_labels,output=False,f=f,training_data=training_data)
        print(f'Overall F1-Score (Test {i}): {Report['macro avg']['f1-score']}',file=f)
        scores.append(Report['macro avg']['f1-score'])
    print('\n',file=f)
    avg_score = sum(scores)/len(scores)

    print(f'Average F1-Score ({i} Tests): {avg_score}',file=f)
    print('-------------------END of RANDOM OTL SELECTION---------------------------',file=f)
    return avg_score

################################OTL selection##############################################
# This is the main function that is used to run all OTL selection methods. The script passes the dataframe "mgb_df" that holds the results from
# detectable subset generation. We execute OTL selection methods such as "mcp", "high_eta", and "random_otl". After finding our OTLs we use
# our "training data" to run KNN classifier considering "k" neighbors for all possible outages in "outage_set". The KNN results are saved in text file "f".
# The main script varies the number of OTLs that are selected ("OTL_num") [1,2,4,8,Full Converage]. 
# OTL seleciton methods
# -MCP: maximum coverage problem, in iterative method to find OTLs that cover the most outages based off their detectable subsets.
#       Runs KNN classifier for all converged outages
# -High Eta: A method that selects OTLs based on the size of our detectable subsets. The larger subsets are considered first.
#            Runs KNN classifier fo all converged outages.
# -Random OTL: A method that selects OTLs at random, the "LOIF" matrix is used to select a random sample from all OTLs.
#              Runs KNN classifier for all converged outages 10 times and gets the average ovrall F1-Score.
#
# Parameters
# - mgb_df: This dataframe holds the results from function "lod_detectable_subsetgen", used in MCP and HIGH ETA OTL selection methods
# - training_data: This is the labeled power flow data obtained from MATLAB script "lod_labeled_datagen.m"
# - outage_set: Set of all outages that converged.
# - k: Number of Neighbors to consider in KNN
# - f: this is the text file where we save our KNN results
# - OTL_num: Number of OTLs to select (From main [1,2,4,8,Full Coverage])
# - LOIF: This the LOIF matrix of the system exlcuding data from outages that don't converge. Used in High Eta Method
def lod_otl_select(mgb_df,training_data,outage_set,k,f,OTL_num,LOIF):
    if OTL_num == "FC":  #If we are performing Full coverage, we set our OTL num to a large number so we select enough lines to cover all outages
        OTL_num = 30000

    Results = {} #Dictionary to store results for each OTL selectio method.
    print(f'Final Sa Subsets:\n {mgb_df}',file=f)
    print('\n',file=f)
    Results['MCP'] = mcp(k=k,mgb_df=mgb_df.sort_values(by='Total_outages',ascending=False),OTL_num=OTL_num,f=f,training_data=training_data,outage_set=outage_set)
    print('\n',file=f)
    Results['Eta'] = high_eta(k=k,mgb_df=mgb_df.sort_values(by='Total_outages',ascending=False),MCP= Results['MCP'],f=f,training_data=training_data,outage_set=outage_set)
    print('\n',file=f)
    Results['Random'] = random_otl(k=k,MCP=Results['MCP'],f=f,training_data=training_data,LOIF=LOIF,outage_set=outage_set)
    
    return Results