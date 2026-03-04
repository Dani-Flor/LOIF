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


##MAIN
def main(data_dir,data_label,system,beta='MIN',gamma='MIN',sol_type='AC',k=6,desired_precision=1.0,desired_f1=1.0):
    start_time = time.perf_counter()

    #Select Which Files to Use
    current_path = os.getcwd()
    folder_path = data_dir
    os.chdir(folder_path)
    matrix_file = f"LOIFmatrix_{system}.csv"
    LOIF = pd.read_csv(matrix_file,index_col=-1)  #Excluded lines are not removed in matlab
    matrix = "LOIF"
    print(LOIF)
    
    td_file = f"{data_label}_trainingdata_{system}_{sol_type}.csv"
    Training_all = pd.read_csv(td_file)
    Training_all.index = Training_all['Label']
    print(Training_all)

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
    outage_set = outage_set[1:]
    #Script Tells us how many lines did not converge, which are the lines we will ignore
    excluded_set = Convergence_Data.loc[Convergence_Data[1] ==0].index.to_list()


    #Add row labels to LOIF matrix
    line_numbers = [i for i in range(1,num_lines+1)]
    LOIF.index = line_numbers
    LOIF.columns = line_numbers
    LOIF = LOIF.drop(index=excluded_set,columns=excluded_set)  #Remove Rows and Columns related to excluded lines

    lod_detectable_subsetgen(gamma=gamma,beta=beta,LOIF=LOIF)
    # for OTL_num in [1,2,4,8,'FC']:
    #     os.chdir(folder_path)
    #     if beta == 'MIN' and gamma == 'MIN':
    #         gb = 'MIN'
    #         f = open(f"output_{OTL_num}OTL_k{k}_{system}_{data_label}_{sol_type}_{matrix}_{gb}_gb.txt", 'w')
    #     else:
    #         gb = 'FIXED'
    #         f = open(f"output_{OTL_num}OTL_k{k}_{system}_{data_label}_{sol_type}_{matrix}_{gb}.txt", 'w')
    #     print(system,file=f)
    #     print(f'DC or AC: {sol_type}',file=f)
    #     print(data_label,file=f)


    #     print(f'Total Number of Transmission Lines: ',num_lines,file=f)
    #     print(f'Possible Line Outages ({len(outage_set)}):\n',outage_set,file=f)
    #     print(f'Excluded Transmission Lines ({len(excluded_set)}):\n',excluded_set,file=f)
        
    #     print(f'================  k-neighbors={k}, Desired Precision(For Normal Conditions)={desired_precision}, Desired F1-Score (Overall Performance)={desired_f1} ================',file=f)
    #     print(f'======================{gb} Approach: Gamma = {gamma}, Beta = {beta}====================',file=f)  
    #   #Checks if minimum gamma results already exist, if not it calculates them first
    #     os.chdir(folder_path)                     #Update path to new folder (Min Gamma Results)
#         if beta == 'MIN' and gamma == 'MIN':
#             if os.path.exists(f'{gb}_gb_k{k}_dp{desired_precision}_df{desired_f1}_{sol_type}.csv'):     #If results for a certain precision and F1-score exist read the CSV file contaiing the reults
#                 mgb_df = pd.read_csv(f'{gb}_gb_k{k}_dp{desired_precision}_df{desired_f1}_{sol_type}.csv',index_col=0)  #Read CSV file
#                 print(mgb_df,file=f)
#             else:
#                 mgb_df = lod_detectable_subsetgen(k=k,gamma=gamma,beta=beta,desired_f1score=desired_f1,desired_precision=desired_precision,LOIF=LOIF,f=f,Training_all=Training_all)
#                 mgb_df.to_csv(f'{gb}_gb_k{k}_dp{desired_precision}_df{desired_f1}_{sol_type}.csv')
#                 mgb_df = pd.read_csv(f'{gb}_gb_k{k}_dp{desired_precision}_df{desired_f1}_{sol_type}.csv',index_col=0)  #Read CSV file
#                 print(mgb_df,file=f)
#         else:
#             if os.path.exists(f'{gb}_{gamma}g_{beta}b_k{k}_{sol_type}.csv'):     #If results for a certain precision and F1-score exist read the CSV file contaiing the reults
#                 mgb_df = pd.read_csv(f'{gb}_{gamma}g_{beta}b_k{k}_{sol_type}.csv',index_col=0)  #Read CSV file
#                 print(mgb_df,file=f)
#             else:
#                 mgb_df = lod_detectable_subsetgen(k=k,gamma=gamma,beta=beta,desired_f1score=desired_f1,desired_precision=desired_precision,LOIF=LOIF,f=f,Training_all=Training_all)
#                 mgb_df.to_csv(f'{gb}_{gamma}g_{beta}b_k{k}_{sol_type}.csv')
#                 mgb_df = pd.read_csv(f'{gb}_{gamma}g_{beta}b_k{k}_{sol_type}.csv',index_col=0)  #Read CSV file
#                 print(mgb_df,file=f)
        
#         lod_otl_select(mgb_df=mgb_df,f=f,Training_all=Training_all,LOIF=LOIF,OTL_num=OTL_num)
#     print('===================Code Executed============================',file=f)
#     end_time = time.perf_counter()
#     elapsed_time = end_time - start_time
#     print(f"Code executed in {elapsed_time:0.4f} Seconds",file=f)
#     elapsed_time = elapsed_time/60
#     print(f"Code executed in {elapsed_time:0.4f} Minutes",file=f)
#     elapsed_time = elapsed_time/60
#     print(f"Code executed in {elapsed_time:0.4f} Hours",file=f)

#     f.close()

# # main(k=6,beta=0.1,gamma=0.1,desired_f1=1.0,desired_precision=1.0)
main(data_dir=r"C:\Users\dflores\Documents\Python\Total_coverage_tests\case118",data_label='demo',system='case118',beta=0.1,gamma=0.1)

