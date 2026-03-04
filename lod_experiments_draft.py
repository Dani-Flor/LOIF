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
from lod_exp_otl_select import *
import matplotlib.pyplot as plt
import seaborn as sns

############# Plot Test Results ###########################
def plot_data(system,gb,test_results,style=1):
    if gb == 'MIN':
        string = "MINIMUM"
    elif gb == 'FIXED':
        string = "FIXED"
    if style == 1:
        otl_levels = test_results.index.unique()

        fig, axes = plt.subplots(1, len(otl_levels), figsize=(24, 8), sharey=True)

        for ax, otl in zip(axes, otl_levels):

            df = test_results.loc[otl]

            df_long = df[["MCP", "High Eta", "Random"]].melt(
                var_name="Method",
                value_name="F1-score"
            )

            sns.pointplot(
                data=df_long,
                x="Method",
                y="F1-score",
                errorbar=("ci", 95),
                capsize=0.15,
                join=False,
                ax=ax
            )

            ax.set_title(otl, fontsize=15)
            ax.set_xlabel("Method", fontsize=20)
            ax.tick_params(labelsize=15)
            ax.grid(True, axis="y")

            ax.set_ylim(0, 1.0)   # ← Added this line

        axes[0].set_ylabel("F1-score", fontsize=20)

        fig.suptitle(f"{system} – Method Performance with 95% CI ({string} Approach)", fontsize=24)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()
    if style == 2:
        # ---------------------------------
        # Prepare dataframe
        # ---------------------------------

        # Reset index so OTL becomes a column
        summary = test_results.reset_index().rename(columns={"index": "OTL"})

        # Convert wide format to long format
        summary_long = summary.melt(
            id_vars=["OTL", "DL"],
            value_vars=["MCP", "High Eta", "Random"],
            var_name="Method",
            value_name="F1-score"
        )

        # Optional: ensure correct OTL order
        otl_order = ["1 OTL", "2 OTL", "4 OTL", "8 OTL", "FC OTL"]
        summary_long["OTL"] = pd.Categorical(summary_long["OTL"], categories=otl_order, ordered=True)

        # ---------------------------------
        # Plot
        # ---------------------------------

        plt.figure(figsize=(10, 6))

        sns.lineplot(
            data=summary_long,
            x="OTL",
            y="F1-score",
            hue="Method",
            errorbar=("ci", 95),
            marker="o"
        )

        plt.ylim(0, 1.0)
        plt.ylabel("F1-score")
        plt.xlabel("OTL Level")
        plt.title(f"{system} – Method Performance vs OTL Level (95% CI) {string} approach")
        plt.grid(True, axis="y")
        plt.legend(title="Method")
        plt.tight_layout()
        plt.show()
    if style == 3:
        # ---------------------------------
        # Prepare dataframe
        # ---------------------------------

        summary = test_results.reset_index().rename(columns={"index": "OTL"})

        summary_long = summary.melt(
            id_vars=["OTL", "DL"],
            value_vars=["MCP", "High Eta", "Random"],
            var_name="Method",
            value_name="F1-score"
        )

        # Optional: enforce OTL order
        otl_order = ["1 OTL", "2 OTL", "4 OTL", "8 OTL", "FC OTL"]
        summary_long["OTL"] = pd.Categorical(summary_long["OTL"], categories=otl_order, ordered=True)

        # ---------------------------------
        # Plot
        # ---------------------------------

        fig, axes = plt.subplots(1, 5, figsize=(24, 8), sharey=True)

        for ax, otl in zip(axes, otl_order):

            df = summary_long[summary_long["OTL"] == otl]

            # Individual test runs
            sns.stripplot(
                data=df,
                x="Method",
                y="F1-score",
                jitter=True,
                alpha=0.5,
                color="black",
                ax=ax
            )

            # Mean + 95% CI
            sns.pointplot(
                data=df,
                x="Method",
                y="F1-score",
                errorbar=("ci", 95),
                join=False,
                capsize=0.15,
                color="red",
                ax=ax
            )

            ax.set_title(otl, fontsize=15)
            ax.set_xlabel("Method", fontsize=20)
            ax.set_ylim(0, 1.0)
            ax.grid(True, axis="y")
            ax.tick_params(labelsize=15)

        axes[0].set_ylabel("F1-score", fontsize=20)

        fig.suptitle(f"{system} – Method Performance with 95% CI (All Test Runs Shown) {string} approach", fontsize=24)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()
    if style == 4:
        # ---------------------------------
        # Prepare dataframe
        # ---------------------------------

        summary = test_results.reset_index().rename(columns={"index": "OTL"})

        otl_order = ["1 OTL", "2 OTL", "4 OTL", "8 OTL", "FC OTL"]
        summary["OTL"] = pd.Categorical(summary["OTL"], categories=otl_order, ordered=True)

        methods = ["MCP", "High Eta", "Random"]

        # ---------------------------------
        # Plot
        # ---------------------------------

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

        for ax, method in zip(axes, methods):

            # Plot each individual test run (thin, transparent lines)
            for dl in summary["DL"].unique():
                subset = summary[summary["DL"] == dl]
                ax.plot(
                    subset["OTL"],
                    subset[method],
                    marker="o",
                    alpha=0.3
                )

            # Overlay mean + 95% CI
            sns.lineplot(
                data=summary,
                x="OTL",
                y=method,
                errorbar=("ci", 95),
                marker="o",
                linewidth=3,
                ax=ax,
                label="Mean ± 95% CI"
            )

            ax.set_title(method, fontsize=16)
            ax.set_xlabel("OTL Level", fontsize=14)
            ax.set_ylim(0, 1.0)
            ax.grid(True, axis="y")

        axes[0].set_ylabel("F1-score", fontsize=14)

        fig.suptitle(f"{system} – Spaghetti Plot (All Test Runs + 95% CI) {string} approach", fontsize=20)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()
# From our previous work, we evaluated each OTL by finding a set of detectable outages with the use of LOIFs.
# LOIF measures the change in power flow of an OTL due to the outage of another in a system.
# In order to find detectable outages using LOIF, we previously implemented a filtering method where each outage in our detectable set (Sa)
# must have an LOIF that satisfies the following thresholds
#       1) LOIF >= gamma  : We want to get rid of outages that will have little impact to the OTL (power flow does not change after outage/ power redistribution).
#    After the first stage of filtering we then have a set of outages that will have a notciable change in power flow at the OTL.
#    From this smaller set (Sa) of outages, we then look for outages that will have different impacts on the OTL. The change in power flow at the OTL should be different for each outage.
#    To do this we find the differences in LOIFs for each outage, and select outages whose LOIF's are seperated by a minimum distance of beta 
#       2) mindist(LOIF) >= beta : We find well-separated LOIFs, by sorting all LOIFs in descending order and finding the differences in consecutive values abs(Sa[i] - Sa[i-1]).
#       With these diferences we select LOIFs whose difference is greater than or equal to beta.
# With these two thresholds we find detectable outage set (Sa). This set of outages should be easily classified with our ML model when only looking at the power flow of the OTL.

# In our previous work we wanted to see the effect of different gamma and beta values and we varied each variable by [0.1, 0.4, 0.8].
#  With Sa we created a observability metric, eta, which we used to select a set of OTLs of sizes [2, 4, 8]. 
#  Because eta is directly influenced by gamma and beta, the OTLs that are selected might change which can affect ML performance. 
# In order to find the best set of detectable outages, we would like to find the optimal values of gamma and beta for each OTL that will result in good ML performance.

# There are two classification measures that we will use to find the best gamma and beta:
#   - Precision: True Positives / (True Positves + False Positives)
#       True Positives: The number samples of a particular class (outage) that the ML model predicted correctly
#       False Positives: The number of samples that were incorrectly predicted as the class (outage) by the ML model.
#      **IF Precision is 1.0, then we know that there are no False Positives meaning that all the samples of the particular class were predicted correctly.
#   -Avg. F1-Score: The harmonic mean of precision and recall, in other words, overall performance metric.
#      **IF Avg. F1-Score is 1.0, then we know that we got 100% accuracy on classification of all labels (outages).

# In order to find the minimum/optimal gamma, we use precision specifically for the class representing normal conditions (0). 
#    - If precision for normal conditions is 1.0, then we know that the outages in the detectable outage set (Sa) provides a significant change in power flow on the OTL 
#     (enough not to be confused with the power flow during normal conditions)
#    - This is done through an iterative process, where we first start with a very low gamma and increase by small increments with each iteration. 
#      In each iteration we find Sa using the current gamma and run KNN classifier as our ML model to collect a classification report.
#      From this report, we observe the precision for normal conditions and check if it meets desired value of 1.0 (desired precision)
#      Once we reach desired precision, we found our optimal/minimum gamma for OTL. 

# In order to find the minimum/optimal best, we use Avg. F1-Score (overall peformance) for all classes.
#    -If Avg. F1-Score is 1.0, then we know that the outages in Sa have LOIFs that are separated from each other by a minimum distance of beta and all outages are predicted correctly.
#    -This is done through an iterative process, where we first start with a very low beta and increase by small increments with each iteration.
#     In each iteration, we find Sa using the current beta and run KNN classifier as our ML model to collect a classification report.
#     From this report, we observe the Avg. F1-Score and check if it meets desired value of 1.0 (desire f1-score)
#     Once we reach desired f1-score, we found our optimal/minimum beta for OTL.




############################ Find Dectectable subsets (lod_detectable_subsetgen)============================================
#This function uses "gamma" and "beta" along with the "LOIF" matrix to find subsets of detectable outage lines.
# We observe each line in the power system and filter out LOIF values that don't satisfy our threshold conditions
# With these subsets we use our "training_data" and call the function "lod_exp_exec" to execute KNN classifier with
# "k" neighbors. We save our results in the the "f" file.
# Users have the option to find minimum beta and gamma through an exhaustive search where we find values that provide a
#  "desried precision" (beta) and a "desired f1-score" (gamma) for each observed line's detectable outage set.
# Users can also provide their own gamma and beta values to find detectable subsets for each observed line.
# 
# Variables
# - beta: Used to filter
# - gamma: Used to filter LOIF values that are small to point where they are not detectable
#        when using the power flow measurements of the observed line 
# - LOIF: This the LOIF matrix of the system exlcuding data from outages that don't converge.
# - training_data: This is the labeled power flow data obtained from MATLAB script "lod_labeled_datagen.m"
# - k: Number of Neighbors to consider in KNN
# - f: this is the text file where we save our KNN results
# - desired_precision: Used to find the minimum beta for each observed line
# - desired_f1score: Used to find the minimum gamma for each observed line
#After finding the subsets, we use run KNN classifier for each subset.
def lod_detectable_subsetgen(beta,gamma,LOIF):
    print('Finding Sa Subsets......')
    Sa_subsets = []  #Append Sa subsets into a list of lists
    L = LOIF.index
    for i in L:
        Sa = LOIF.loc[i,:]
        Sa = Sa[abs(Sa) >= beta]
        Sa = Sa.sort_values(ascending=False)
        #Find LOIFs whose abs. differences are greater than beta, provides LOIFs that are separated by a distance of beta
        Sa = Sa[Sa.diff().abs().fillna(0) >= gamma]
        # print(Sa)
        Sa_subsets.append(Sa.index.to_list())   #get the corresponding line numbers of remaining LOIF values after beta and gamma filtering
    print(Sa_subsets)
    return Sa_subsets
# def lod_detectable_subsetgen(beta,gamma,LOIF,training_data,k,f,desired_precision,desired_f1score):
#     print('Finding Sa Subsets......')

#     if beta == 'MIN' and gamma == "MIN":
#         #
#         final_sets = []   #Will append every Sa subset after finding minimum values
#         set_lenghts = []  #Will append the size of every final Sa subset of each observed transmission line
#         min_gammas = []   #Will append the minimum gamma for each observed transmisson line
#         min_betas  = []   #Will append the minimum beta for each observed transmission line
#         final_f1s = []    #Will append the avg. f1 scores for each observed transmsission line
#         index_col = []    #Will append the line number for each observed transmission line
#         for i in LOIF.index:
#             # Ta = Ta_sets[i]      #Ta will hold the LOIFs of all outages for current OTL i (grab from dict.)   #create pandas serires Sa (initially holds LOIFs of all outages)

#             gamma = 1e-30       #start with low gamma
#             beta = 1e-30        #start with low beta

#             #pull columns from Training Data that are needed (Power Flows of OTL + Label)
#             #stops from having to access whole dataset.
#             otl_col = []        #column labels of OTL to filter Training Data 
#             otl_col.append(f'PF Line {i}')  #Active Power at From Bus
#             otl_col.append(f'QF Line {i}')  #Reactive Power at From Bus
#             otl_col.append(f'PT Line {i}')  #Active Power at To Bus
#             otl_col.append(f'QT Line {i}')  #Reactive Power at To Bus
#             otl_col.append(f'Label')        #Class labels

#             #Find Min. Gamma
#             current_precision = 0.0  #intialize current precision with zero (will update with each iteration)
#             while(1): #Keep looping until minimum gamma is found
#                 Sa = pd.Series(LOIF.loc[i,:])              #create pandas serires Sa (initially holds LOIFs of all outages)
#                 Sa = Sa[abs(Sa) >= beta]       #Find LOIFs greater than gamma
        
#                 #Get the line numbers of the outages that satisfied current gamma condition.
#                 outage_labels = [i] + Sa.index.to_list()   #Always include the line number of the OTL (Easy to detect, since no power should be flowing through OTL)
#                 outage_labels = list(set(outage_labels))   #used to pull the rows from Training Data for the following classes

#                 #Runn KNN classifier (Make Sure to include normal condtions, 0, in outage_labels)
#                 Report = lod_exp_exec(otl_col=otl_col,outage_labels=[0] + outage_labels,k=k,output=False,f=f,training_data=training_data)
#                 current_precision = round(Report['0']['precision'], 2)    #Collect Precision For Normal Conditions (Label/Class: 0)

#                 #Check if current precision meets the desired precision of 1.0 (or other value depending on user)
#                 #If desired precision is reached OR Sa subset if of OTL hase length of 1, or if gamma passes the value 10 (Loop Stopper, usually means it failed to reach desired precision)
#                 if (current_precision >= desired_precision) or (len(Sa)==1) or (beta == 10):  #If desired precision is reached save min gamma and print classification report to Txt file
#                     minbeta = beta
#                     print(f'Line {i} --> gamma: {minbeta}',file=f)
#                     Report = lod_exp_exec(otl_col=otl_col,outage_labels=[0]+outage_labels,k=k,output=True,f=f,training_data=training_data)
#                     break
#                 elif beta < 0.1:   #If desired precision is not met, and gamma is below 0.1, increase by a factor of 1e1
#                     beta = beta*(1e1)
#                 else:              #If desired precision is not met, and gamma is above 0.1, increase in increments of 0.01
#                     beta = round(beta + 0.01,2)

#             #Find Min. Beta
#             current_f1score = 0.0  #initalize current f1score with zero (will update with each iteration)
#             while(1): #Keep looping until minimum beta is found
#                 Sa = pd.Series(LOIF.loc[i,:])  #create pandas serires Sa (initially holds LOIFs of all outages)
#                 Sa = Sa[abs(Sa) >= minbeta]  #Include min. gamma threshold
#                 Sa = Sa.sort_values(ascending=False)  #Organize LOIFs in descending order
                
#                 #Find LOIFs whose abs. differences are greater than beta, provides LOIFs that are separated by a distance of beta
#                 Sa = Sa[Sa.diff().abs().fillna(0) >= gamma]  

#                 outage_labels = [i] + Sa.index.to_list()  #Always include the line number of the OTL (Easy to detect, since no power should be flowing through OTL)
#                 outage_labels = list(set(outage_labels))  #used to pull the rows from Training Data for the following classes
                
#                 #Runn KNN classifier (Make Sure to include normal condtions, 0, in outage_labels)
#                 Report = lod_exp_exec(otl_col=otl_col,outage_labels=[0] + outage_labels,k=k,output=False,f=f,training_data=training_data)
#                 current_f1score = Report['macro avg']['f1-score']       #Collect Avg. F1-Score (Overall Performance)
                
#                 #Check if current f1-score meets the desired f1-score of 1.0 (or other value depending on user)
#                 #If desired f1-score is reached OR length of Sa is 1 OR if beta passes the value 10 (Loop Stopper, usually means it failed to reach desired F1-Score)
#                 if (current_f1score >= desired_f1score) or (len(Sa)==1) or (gamma == 10): #If desired f1-score is reached save min beta and print classification report to Txt file
#                     mingamma = gamma
#                     print(f'Line {i} --> beta: {minbeta}',file=f)
#                     Report = lod_exp_exec(otl_col=otl_col,outage_labels=[0] + outage_labels,k=k,output=True,f=f,training_data=training_data)
#                     final_f1s.append(current_f1score)
#                     break
#                 elif gamma < 0.1:   #If desired f1-score is not met, and beta is below 0.1, increase by a factor of 1e1
#                     gamma = gamma*(1e1)
#                 else:              #If desired f1-score is not met, and gamma is above 0.1, increase in increments of 0.01
#                     gamma = round(gamma + 0.01,2) 
        
#             min_gammas.append(mingamma)
#             min_betas.append(minbeta) 
#             set_lenghts.append(len(outage_labels))    
#             final_sets.append(outage_labels)
#             index_col.append(i)

#             #Save Reults for Each OTL into DataFrame and Return DataFrame
#             DF = pd.DataFrame({'Lines':index_col,'min_beta':min_betas,'min_gamma':min_gammas,'Actual_F1Score':final_f1s,'Total_outages':set_lenghts,'Sa_Subsets':final_sets})
#             DF=DF.set_index('Lines')
#     else:
#         betas = []    #Append beta values for DataFrame
#         gammas = []   #Append gamma values for DataFrame
#         final_subsets = [] #Append Sa Subsets (Outage Labels NOT LOIF) for each OTL
#         set_lengths = []   #Append the sizes of each Sa subsets
#         index_col = []  #Append Line Numbers of OTL
#         f1_scores = []  #Appned the F1-Scores of each Sa Subset when running KNN
#         for i in LOIF.index:
#             betas.append(beta)
#             gammas.append(gamma)
#             index_col.append(i)

#             # Ta = Ta_sets[i]
#             #pull columns from Training Data that are needed (Power Flows of OTL + Label)
#             #stops from having to access whole dataset.
#             otl_col = []        #column labels of OTL to filter Training Data 
#             otl_col.append(f'PF Line {i}')  #Active Power at From Bus
#             otl_col.append(f'QF Line {i}')  #Reactive Power at From Bus
#             otl_col.append(f'PT Line {i}')  #Active Power at To Bus
#             otl_col.append(f'QT Line {i}')  #Reactive Power at To Bus
#             otl_col.append(f'Label')

#             Sa = pd.Series(LOIF.loc[i,:])
#             Sa = Sa[abs(Sa) >= beta]
#             Sa = Sa.sort_values(ascending=False)  #Organize LOIFs in descending order
#             #Find LOIFs whose abs. differences are greater than beta, provides LOIFs that are separated by a distance of beta
#             Sa = Sa[Sa.diff().abs().fillna(0) >= gamma]  

#             outage_labels = [i] + Sa.index.to_list()  #Always include the line number of the OTL (Easy to detect, since no power should be flowing through OTL)
#             outage_labels = list(set(outage_labels))  #used to pull the rows from Training Data for the following classes               
#             final_subsets.append(outage_labels)
#             set_lengths.append(len(outage_labels))
#             #Runn KNN classifier (Make Sure to include normal condtions, 0, in outage_labels)
#             Report = lod_exp_exec(otl_col=otl_col,outage_labels=[0] + outage_labels,k=k,output=True,f=f,training_data=training_data)
#             f1_scores.append(Report['macro avg']['f1-score'])
#         DF = pd.DataFrame({'Lines':index_col,'Beta':betas,'Gamma':gammas,'Actual_F1score':f1_scores,'Total_outages':set_lengths,'Sa_Subsets':final_subsets})
#         DF = DF.set_index('Lines')

#     return DF

#####################################MAIN################################################
# This is our main function where we run all other functions. To run functions such as 
# "lod_detectable_subsetgen", "lod_exp_exec", and "lod_otl_select"; users must provide a "file_directory"
# to access important files such as the LOIF matrix, convergence data, and labeled training data that are obtained
# from matlab. The user must specify which "system" (ex. case_ieee30, case118, case_ACTIVSg2000,etc.) they are interested 
# in, and a "data_label" (ex. demo,Test1,Test1,Test3,etc.). With this we can run our experimentation where we find detectable subsets.
# Users can also specify which solution_type ("sol_type") they are intersted in AC/DC power flow solutions.
# subsets for different numbers of observed transmission lines (1, 2, 4, 8, and full coverage). These OTLs are selected with different
# methods such as maximum coverage problem, high eta, and random OTL selection.
# In this script, the user has the option to provide their own "beta" and "gamma" values for "lod_detectable_subsetgen", or by default
# the script will perform an exhaustive search to find MINIMUM gamma and beta values for each OTL. We also use "lod_exp_exec" to run our KNN classifier
# where we consider "k" neighbors, by default k = 6.
# If the user wants to find minimum "beta" and "gamma" values, the user can specify a "desired_precision" and "desired_f1" that min. beta and min. gamma must
# satisfy when finding detectable subsets. By default desired precision and f1-score is set to 1.0.
#
# In addition this script uses a timer to calculate the computation time. Uses convergence data to find the set of outages that converged and did not converge.
# With this information we filter the LOIF matrix to remove data for outages that did not coverge.
# The script also creates an output file where we save our results from "lod_detectable_subsetgen", and creates an ouput text file where we save our KNN classification results.
#
# Required Arguments
# -file_directory: This is the path where our LOIF matrix, convergence data, and training data files are located. (REQUIRED to run script)
# -system: Name of the case system the user is interested in (Ex. case_ieee30, case118, case_ACTIVSg2000, caseWisconsin_1664) (REQUIRED to run script)
# -data_label: This is the label to specify which files of the particular system the user is interested in (Ex. demo, Test1, Test2,etc.) (REQUIRED to run script)
# Optional Arguments
# -beta: This is used to find detectable subsets, filters out LOIFs for outages that will have little to no impact on the observed line.
#        By default beta = 'MIN' meaning it will find minimum beta values.
# -gamma: This is used to find detectable subsets, filters out LOIFs for outages that do not have a minimum distance of gamma from all other LOIFs.
#         By default gamma = 'MIN' meaning it will find minimum gamma values.
#         ***IMPORTANT, both beta and gamma must equal 'MIN' in order to perfrom exhaustive search.***
# -sol_type: AC or DC power flow solutions used in "lod_labeled_datagen.m". By default sol_type = 'AC'
# -k: The number of neigbhors to consider in our KNN classifier. By default k=6
# -desired_precision: Used when finding MINIMUM beta values. By default desired_precision = 1.0
# -desired_f1: Used when finding MINIMUM gamma values. By default desired_f1 = 1.0
def main(data_dir,data_label,system,beta='MIN',gamma='MIN',sol_type='AC',k=6,desired_precision=1.0,desired_f1=1.0):
    start_time = time.perf_counter()

    #Select Which Files to Use
    current_path = os.getcwd()
    folder_path = data_dir
    os.chdir(folder_path)
    matrix_file = f"LOIFmatrix_{system}.csv"
    LOIF = pd.read_csv(matrix_file,index_col=-1)  #Excluded lines are not removed in matlab
    matrix = "LOIF"
    
    td_file = f"{data_label}_trainingdata_{system}_{sol_type}.csv"
    training_data = pd.read_csv(td_file)
    training_data.index = training_data['Label']
    print(training_data)

    cd_file =f"{data_label}_convergence_{system}_{sol_type}.csv"
    Convergence_Data = pd.read_csv(cd_file,index_col=0,header=None)
    print(Convergence_Data)

    
    folder_path = f'{current_path}/Final_Results_{system}_{data_label}_{sol_type}_{matrix}' #Create Path for New Folder to store (Min Gamma Results)
    os.makedirs(folder_path, exist_ok=True)   #If the Folder already exist do nothing, if not then create new folder


    #Script Calculates How Many Transmission Lines There are
    num_lines = len(LOIF.index.to_list())

    #Because We Ignore outages that don't converge when disconnected, we want to see all outages that we collected data on (possible outages)
    #Script Tells us how many lines converged, which are the lines we will focus on.
    outage_set = Convergence_Data.loc[Convergence_Data[1] ==1].index.to_list()
    outage_set = outage_set[1:] #converged outages
    #Script Tells us how many lines did not converge, which are the lines we will ignore
    excluded_set = Convergence_Data.loc[Convergence_Data[1] ==0].index.to_list() #outages that did not converge


    #Add row and column labels to LOIF matrix, and drop data in LOIF regarding outages that did not converge
    line_numbers = [i for i in range(1,num_lines+1)]
    LOIF.index = line_numbers
    LOIF.columns = line_numbers
    LOIF = LOIF.drop(index=excluded_set,columns=excluded_set)  #Remove Rows and Columns related to excluded lines
    print(LOIF)
    f1_dict = {}
    #Repeats 
    # for OTL_num in [1,2,4,8,'FC']:
    #     os.chdir(folder_path)  #change directory to file_directory provided by user
        
    #     #Will create a specific text file depending gamma and beta. (MIN by default)
    #     if beta == 'MIN' and gamma == 'MIN':
    #         gb = 'MIN'
    #         f = open(f"output_{OTL_num}OTL_k{k}_{system}_{data_label}_{sol_type}_{matrix}_{gb}_gb.txt", 'w')
    #     else:
    #         gb = 'FIXED'
    #         f = open(f"output_{OTL_num}OTL_k{k}_{system}_{data_label}_{sol_type}_{matrix}_{gb}.txt", 'w')

    #     #print user parameters and other important information
    #     print(system,file=f)
    #     print(f'DC or AC: {sol_type}',file=f)
    #     print(data_label,file=f)
    #     print(f'Total Number of Transmission Lines: ',num_lines,file=f)
    #     print(f'Possible Line Outages ({len(outage_set)}):\n',outage_set,file=f)
    #     print(f'Excluded Transmission Lines ({len(excluded_set)}):\n',excluded_set,file=f)
    #     print(f'================  k-neighbors={k}, Desired Precision(For Normal Conditions)={desired_precision}, Desired F1-Score (Overall Performance)={desired_f1} ================',file=f)
    #     print(f'======================{gb} Approach: Gamma = {gamma}, Beta = {beta}====================',file=f)  

    lod_detectable_subsetgen(beta=beta,gamma=gamma,LOIF=LOIF)
    #   #Checks if minimum/fixed beta and gamma results (detectable subsets) already exist, if not it calculates them first
    #     if beta == 'MIN' and gamma == 'MIN':  #Used if user wants to perform exhaustive search to find detectable subsets with min. beta and gamma
    #         if os.path.exists(f'{gb}_gb_k{k}_dp{desired_precision}_df{desired_f1}_{sol_type}.csv'):     #If results for a certain precision and F1-score exist read the CSV file contaiing the reults
    #             mgb_df = pd.read_csv(f'{gb}_gb_k{k}_dp{desired_precision}_df{desired_f1}_{sol_type}.csv',index_col=0)  #Read CSV file
    #             print(mgb_df,file=f)
    #         else:
    #             mgb_df = lod_detectable_subsetgen(k=k,gamma=gamma,beta=beta,desired_f1score=desired_f1,desired_precision=desired_precision,LOIF=LOIF,f=f,training_data=training_data)
    #             mgb_df.to_csv(f'{gb}_gb_k{k}_dp{desired_precision}_df{desired_f1}_{sol_type}.csv')
    #             mgb_df = pd.read_csv(f'{gb}_gb_k{k}_dp{desired_precision}_df{desired_f1}_{sol_type}.csv',index_col=0)  #Read CSV file
    #             print(mgb_df,file=f)
    #     else:    #Used to find detectable subsets
    #         if os.path.exists(f'{gb}_{gamma}g_{beta}b_k{k}_{sol_type}.csv'):     #If results for a certain precision and F1-score exist read the CSV file contaiing the reults
    #             mgb_df = pd.read_csv(f'{gb}_{gamma}g_{beta}b_k{k}_{sol_type}.csv',index_col=0)  #Read CSV file
    #             print(mgb_df,file=f)
    #         else:
    #             mgb_df = lod_detectable_subsetgen(k=k,gamma=gamma,beta=beta,desired_f1score=desired_f1,desired_precision=desired_precision,LOIF=LOIF,f=f,training_data=training_data)
    #             mgb_df.to_csv(f'{gb}_{gamma}g_{beta}b_k{k}_{sol_type}.csv')
    #             mgb_df = pd.read_csv(f'{gb}_{gamma}g_{beta}b_k{k}_{sol_type}.csv',index_col=0)  #Read CSV file
    #             print(mgb_df,file=f)
        
    #     #Used to execute various OTL seletion methods (MCP,High Eta, Rand...)
    #     # Results = lod_otl_select(k=k,mgb_df=mgb_df,f=f,training_data=training_data,LOIF=LOIF,OTL_num=OTL_num,outage_set=outage_set)
    #     # print(Results['MCP'][2])
    #     # print(Results['Eta'])
    #     # print(Results['Random'])
    #     # f1_dict[f'{OTL_num} OTL'] = [Results['MCP'][2],Results['Eta'],Results['Random'],data_label]
    #     os.chdir(current_path)
    # #Calculate computation time in seconds, minutes, and hours
    # print('===================Code Executed============================',file=f)
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print(f"Code executed in {elapsed_time:0.4f} Seconds",file=f)
    # elapsed_time = elapsed_time/60
    # print(f"Code executed in {elapsed_time:0.4f} Minutes",file=f)
    # elapsed_time = elapsed_time/60
    # print(f"Code executed in {elapsed_time:0.4f} Hours",file=f)
    
    # # print(pd.DataFrame(f1_dict,index=['MCP','High Eta','Random','Data Label']).transpose())
    # f.close()  #close text file
    # # return pd.DataFrame(f1_dict,index=['MCP','High Eta','Random','DL']).transpose(),system,gb


################Plot Results##################################
# test_results = pd.DataFrame()
# for i in range(1,25):
#     current_test = main(data_dir=r"C:\Users\dflores\Documents\Python\Total_coverage_tests\case_ieee30",data_label=f'Test{i}',system='case_ieee30')#,gamma=0.1,beta=0.1)
#     test_results = pd.concat([test_results,current_test[0]])

# print(test_results)
# plot_data(system=current_test[1],gb=current_test[2],test_results=test_results,style=1)
# plot_data(system=current_test[1],gb=current_test[2],test_results=test_results,style=2)

main(data_dir=r"C:\Users\dflores\Documents\Python\Total_coverage_tests\case118",data_label=f'demo',system='case118',gamma=0.1,beta=0.1)