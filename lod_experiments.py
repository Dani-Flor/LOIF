import pandas as pd
from sklearn.metrics import classification_report  #, f1_score,recall_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
from lod_exp_otl_select import *
from lod_exp_ml import *
from lod_lineplots import *
import numpy as np

############################ Find Dectectable subsets (lod_detectable_subsetgen)============================================
# This function uses "beta", "gamma", and the "LOIF"/"LODF" to find detectable Sa subsets for each OTL in the system.
# Beta filters values that are small, removing outages that have little to no impact
# Gamma filters values that are not separated by a minimum distance, removing outages that have similar impact. 
# Parameters
# beta: Used to filter LOIF/LODF values that are small to point where they are not detectable (outages with little to no change in power flow at the OTL)
# gamma: Used to filter LOIF/LODF values that are not separated by a minimum distance of gamma (outages with similar change in power flow at the OTL)
# LOIF: This the LOIF matrix of the system exlcuding data from outages that don't converge.
# excluded_set: This is the set of outages that did not converge. Used to make sure we don't include any of these outages in ou final sa subsets
#
# Output
# Sa_subsets: a list of lists containing the detectable susbets (Sa) of all OTLs
def lod_detectable_subsetgen(beta,gamma,LOIF,excluded_set):
    print('Finding Sa Subsets......')
    Sa_subsets = []  #Append Sa subsets into a list of lists
    L = LOIF.index
    for i in L:
        Sa = LOIF.loc[i,:]
        Sa = Sa[abs(Sa) >= beta]
        Sa = Sa.sort_values(ascending=False)
        #Find LOIFs whose abs. differences are greater than beta, provides LOIFs that are separated by a distance of beta
        Sa = Sa[Sa.diff().abs().fillna(0) >= gamma]
        Sa = Sa.index.to_list()
        if (i not in Sa) and (i not in excluded_set):
            Sa.append(i)     #Include the observed line in outage set, since it is obvious based on power flow that it is out.
        Sa_subsets.append(Sa) 
    
    return Sa_subsets

start_time = time.perf_counter()

###################### LOD Experiment ################################################
#                        Parameters
# data_dir: This is the folder path containing you important files needed to run to code such as the
#            LOIF/LODF matrix, convergence data, and training data
#
# data_label: This is the file label the user is interested in. Ex. "demo", "Test1", "Test2", etc.
#
# system: This is the name of the case system the use is interested in. Ex. 'case_ieee30','case118','case_ACTIVSg2000','caseWisconsin_1664'
# # Optional Parameters
# beta: This is used in the function "lod_detectable_subsetgen" to filter LOIF/LODF values that are small (little to no change in power flow on OTL)
#       Default = 0.1
# gamma: This is used in the function "lod_dectable_subsetgen" to filter LOIF/LODF values that are not separated by a minimum distance of beta to all other LOIFs.
#        Default = 0.1
# sol_type: The user has the option to use DC or AC power flow measurements which are obtained in "lod_labeled_datagen.m"
#           Default = 'AC'
# k: The number of neighbors to consider in our KNN classification model in the function "lod_exp_exec"
#    Default = 6
# matrix: The user has the option to use LOIF or LODF matrices
#         Default = 'LOIF'
# 
# OUTPUT
# all_results: This is a 3x5 dataframe that holds the f1scores of our MCP, High Eta, and Random OTL selection for 1,2,4,8 OTLs, and Full Coverage
# system: This is used in to create the title of our plots, shows which system the results belong to.
# matrix: This is used to create the title of our plots, shows which matrix was used LOIF/LODF.
def main(data_dir,data_label,system,beta=0.1,gamma=0.1,sol_type='AC',k=6,matrix='LOIF'):
    #Select Which Files to Use
    current_path = os.getcwd()
    folder_path = data_dir
    os.chdir(folder_path)
    matrix_file = f"{matrix}matrix_{system}.csv"
    LOIF = pd.read_csv(matrix_file,index_col=-1)  #Excluded lines are not removed in matlab
    
    td_file = f"{data_label}_trainingdata_{system}_{sol_type}.csv"
    training_data = pd.read_csv(td_file)
    training_data.index = training_data['Label']
    print(training_data)

    cd_file =f"{data_label}_convergence_{system}_{sol_type}.csv"
    Convergence_Data = pd.read_csv(cd_file,index_col=0,header=None)

    f = open(f"output_{system}_{sol_type}_{matrix}_{data_label}.txt", 'w')

    #Calculates How Many Transmission Lines There are
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
    LOIF.loc[:,excluded_set] = np.nan
    # print(LOIF)

    ################################ Detectable Subset Generation ################################
    Sa_subsets = lod_detectable_subsetgen(beta=beta,gamma=gamma,LOIF=LOIF,excluded_set=excluded_set)   #list of lists
    Sa_lengths = [len(i) for i in Sa_subsets]
    dict = {}
    dict['Sa_Subsets'] = Sa_subsets
    dict['Total_Outages'] = Sa_lengths
    Sa_df = pd.DataFrame(dict,index=LOIF.index)

    print(Sa_df,file=f)

    all_results = pd.DataFrame()

    for OTL_num in [1,2,4,8,'FC']:
        print(f'--------------- Number of OTLs: {OTL_num} ---------------------',file=f)

        all_f1_results = []
        # all_recall_results = []
        # all_precision_results = []
        ################################### OTL Selection ############################################
        mcp_otls = mcp(df=Sa_df,OTL_num=OTL_num)
        print(f'MCP selects the following OTLs ({len(mcp_otls)}): {mcp_otls}\n',file=f)
        he_otls = high_eta(MCP=mcp_otls,df=Sa_df)
        print(f'HE selects the following OTLs ({len(he_otls)}): {he_otls}\n',file=f)
        rand_otls = random_otl(MCP=mcp_otls,LOIF=LOIF)
        print(f'Random selects the following OTLs ({len(rand_otls)}): {rand_otls}',file=f)
        ############################ LOD Experiment Execution (ML Classification) ############################
        print("Calculating MCP OTL results...")
        mcp_results = lod_exp_exec(otl_set=mcp_otls,outage_set=outage_set,k=k,training_data=training_data,output=False)
        all_f1_results.append(mcp_results['macro avg']['f1-score'])
        # all_recall_results.append(mcp_results['macro avg']['recall'])
        # all_precision_results.append(mcp_results['macro avg']['precision'])
        print(f'MCP Classification Results: {mcp_results}\n',file=f)

        print("Calculating High Eta OTL results....")
        he_results = lod_exp_exec(otl_set=he_otls,outage_set=outage_set,k=k,training_data=training_data,output=False)
        all_f1_results.append(he_results['macro avg']['f1-score'])
        # all_recall_results.append(he_results['macro avg']['recall'])
        # all_precision_results.append(he_results['macro avg']['precision'])
        print(f'HE Classification Results: {he_results}\n',file=f)

        print("Calculating Random OTL results (Average of 10 tests).....")
        rand_f1 = []
        rand_recall = []
        rand_precision = []
        for i in rand_otls:
            current_result = lod_exp_exec(otl_set=i,outage_set=outage_set,k=k,training_data=training_data,output=False)
            rand_f1.append(current_result['macro avg']['f1-score'])
            rand_recall.append(current_result['macro avg']['recall'])
            rand_precision.append(current_result['macro avg']['precision'])


        all_f1_results.append(sum(rand_f1)/len(rand_f1))
        print(f'Random OTL Average F1-Score: {sum(rand_f1)/len(rand_f1)}',file=f)
        # all_recall_results.append(sum(rand_recall)/len(rand_recall))
        print(f'Random OTL Average F1-Score: {sum(rand_recall)/len(rand_recall)}',file=f)
        # all_precision_results.append(sum(rand_precision)/len(rand_precision))
        print(f'Random OTL Average F1-Score: {sum(rand_precision)/len(rand_precision)}',file=f)

        all_results[OTL_num] = all_f1_results

    all_results['DL'] = [data_label for i in range(1,4)]
    all_results.index = ['MCP','HE','Rand']




    return all_results,system,matrix
    
## Collect results for all tests
## If the user has data labels in sequential order (Ex. Test1,Test2,Test3,.....,Test24), they can use the following lines of code to collect data for all tests
## We use a for loop to call main() repeatedly where we are changing the datalabel being used. This can be looped any number of times depending on the number
## Tests the user is interested. 
## The results in main are then concatenated into one dataframe which is then used to create our plots.
results = pd.DataFrame()
for i in range(1,25):
    current_test = main(data_dir=r"C:\Users\dflores\Documents\Python\Total_coverage_tests\case118",data_label=f'Test{i}',system='case118',k=6,matrix='LODF')
    results = pd.concat([results,current_test[0]])
print(results)

#Calculate computation time in seconds, minutes, and hours
print('===================Code Executed============================')
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Code executed in {elapsed_time:0.4f} Seconds")
elapsed_time = elapsed_time/60
print(f"Code executed in {elapsed_time:0.4f} Minutes")
elapsed_time = elapsed_time/60
print(f"Code executed in {elapsed_time:0.4f} Hours")

plot_data(system=current_test[1],matrix=current_test[2],test_results=results,style=1)