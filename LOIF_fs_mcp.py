#Daniel Flores (UMASS), Michael P. McGarry (UTEP), Yuanrui Sang (UMASS)
# Feature Selection Tool and Maximum Coverage Problem (MCP-Greedy)
# There are three files that are needed in order to run this script
#       -Branch Data CSV: MATPOWER Outage Simulations of Power System
#           > From Matlab, these files have names starting with "Branchdata_..."
#       -Convergence Data CSV: MATPOWER Convergence Results For Outage Simulations
#           > From Matlab, these files have names starting with "Convergence_Data..."
#           > This file tells us which outages converged successfully and which ones didn't
#           > We only focus on outages that converged
#       -LOIF Matrix CSV: Holds the LOIFs (Impact of each outage on all lines)
#           > From Matlab, these files have names starting with "LOIFmatrix_..." 
# This script will open a directory window and have the user select which files they want to use (READ Window Title to Know which File Python Wants)
# *REMARK: Make Sure the System you are using is the same on all files (If you accidently select files from another system you will get an error)
#
#  The user also has to specify whether the training data uses AC or DC solutions
##################################  Script Functions ##############################################################################
#  -select_files(t): 
#   This function opens a file directory and asks the user for required csv files
#       -t : Type of File User needs to select
#           > t = 1 --> Script ask user to select Convergence Data  CSV File
#           > t = 2 --> Script asks user to select Branch Data CSV file. Script May take longer to open for larger systems.
#           > t = 3 --> Script ask user to select LOIF Matrix CSV file.
#
#        -RETURNS: path of selected file
#
#  -knn_model(ML_labels,OL_labels,k): 
#   This Function Filters Training Data Depending on the Observe Transmission Lines Selected and the Outages we want to test.
#       -ML_labels : Set of Observed Transmission Lines (OTL) to get power flow data from.
#       -OL_labels : Set of Outages We Want to Test with the following power flow data of OTLs
#       -k : The Number of K-Neighbors in KNN Classifier
#
#       -RETURNS: Classification Report of KNN Classifier (In Dictionary Form)
#
#  -calc_min_gamma_beta(k,desired_precision,desired_f1score): 
#   This Function Will Cycle Through Each OTL and Finds the best Sa Subset by finding
#   the minimum gamma and minimum beta through the use of desired performance metrics. 
#   sa = {x ∈ Ta||x| ≥ γ, mindist(x) ≥ β}. Each abs. value of each LOIF must be greater than gamma. Each LOIF remaining must have a min. distance >= beta from each other.
#       -k : The Number of K-Neighbors in KNN Classifier. This function calls knn_model() multiple times.
#       -desired_precision : The Precision (Specifically for Normal Conditions: 0) That Each Sa Subset Must Satisfy In order to reach minimum gamma.
#       -desired_f1score : The F1-Score (Overall Performance) that each Sa subset must satisfy in order to reach minimum beta
#
#       -RETURNS: Dataframe containing information on min. gamma, min. beta, Sa subset, Length of Sa subset (Total Outages), and the Actaul F1-Score that satisfied desired F1-Score Condition
#
#  -max_coverage_greedy(mgb_df,OTL_num):
#   This Function Will Execute the MCP algorithm after finding the Sa subsets with their min. gamma and min. beta for each OTL.
#       -mgb_df: This is the dataframe created from calc_min_gamma_beta(). Used for MCP Alorithm
#       -OTL_num: The number of OTLs we want to get max coverage with.
#           > Right Now this is set to a Very Large Number In Order find the maximum number of OTLs needed for max coverage
#
#  -main(k,desired_precision,desired_f1):
#   This is the function that brings everything together and is the only function that needs to be called. This function also creates a directory to save newly created Dataframes.
#   If this function was already ran once, and dataframes already exist. This script will only run max_coverage_greedy().
#       -k : The Number of K-Neighbors in KNN Classifier specified by user. Used in calc_min_gamma_beta()
#       -desired_precision : The Precision (Specifically for Normal Conditions: 0) specified by user. Used in calc_min_gamma_beta()
#       -desired_f1score :   The F1-Score (Overall Performance) specified by user. Used in calc_min_gamma_beta()
############################### Summary #################################################################
# This script will create Sa subsets for each OTL in the system by first
#    -Finding minimum gamma for each Sa subset
#          *User specifies a desired Precision (For Normal conditions) in order to find gamma.
#          *Precision tells us if the KNN model cofuses outages with normal conditions (If set to 1.0, no confusion) when testing each Sa subset  
#    -Finding minimum beta for each Sa subset
#          *User specifies a desired F1-Score (Overal Performance) in order to find beta
#          *F1-score tells us how well the KNN model should perform when testing each Sa subset
# After finding the minimum gamma and beta values for each Sa subset, this script will run a Greedy Maximum Coverage Problem in order to find the maximum coverage with a set of OTLs

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

start_time = time.perf_counter()
###USER INPUTS#######
#Read Bracnh Data CSV collected from MATLAB using MATPOWER
root = tk.Tk()
root.withdraw() 
case = ''
def select_files(t):
    if t == 1:
        title = "SELECT CONVERGENCE FILE OF THE SYSTEM YOU ARE INTERESTED IN"
    elif t == 2:
        title = "SELECT BRANCH DATA FILE OF THE SYSTEM YOU ARE INTERESTED IN (MAKE SURE IT IS THE SAME SYSTEM YOU INITIALLY CHOSE)"
    elif t == 3:
        title = 'SELECT LOIF MATRIX FILE OF THE SYSTEM YOU ARE INTERESTED IN (MAKE SURE IT IS THE SAME SYSTEM YOU INITIALLY CHOSE)'
    fpath = filedialog.askopenfilename(
        title=title,
        initialdir="/", # Optional: sets the initial directory
        filetypes=(
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        )
    )
    print(fpath)
    return fpath

#Select Which Files to Use
fpath = select_files(1)
Convergence_Data = pd.read_csv(fpath,index_col=0,header=None)
fpath = select_files(3)
LOIF = pd.read_csv(fpath)  #Excluded lines are not removed in matlab
case = re.search(r"LOIFmatrix_(case.+)\.csv", fpath).group(1)
print(case)
fpath = select_files(2)
Training_all = pd.read_csv(fpath)
Training_all.index = Training_all['Label']

DC_or_AC = 'DC'
####User Inputs
it = 0

#Script Calculates How Many Transmission Lines There are
num_lines = len(LOIF.index.to_list())
print(f'Total Number of Transmission Lines: ',num_lines)
#Because We Ignore outages that don't converge when disconnected, we want to see all outages that we collected data on (possible outages)
#Script Tells us how many lines converged, which are the lines we will focus on.
outage_set = Convergence_Data.loc[Convergence_Data[1] ==1].index.to_list()
outage_set = outage_set[1:]
#Script Tells us how many lines did not converge, which are the lines we will ignore
excluded_set = Convergence_Data.loc[Convergence_Data[1] ==0].index.to_list()
print(f'Possible Line Outages ({len(outage_set)}):\n',outage_set)
print(f'Excluded Transmission Lines ({len(excluded_set)}):\n',excluded_set)


#Add row labels to LOIF matrix
line_numbers = [i for i in range(1,num_lines+1)]
LOIF.index = line_numbers
LOIF.columns = line_numbers
LOIF = LOIF.drop(index=excluded_set,columns=excluded_set)  #Remove Rows and Columns related to excluded lines


current_path = os.getcwd()   #Get the address of current path
folder_path = f'{current_path}/Final_Sa_{case}_{DC_or_AC}' #Create Path for New Folder to store (Min Gamma Results)
os.makedirs(folder_path, exist_ok=True)   #If the Folder already exist do nothing, if not then create new folder


Ta_sets = {}
for i in LOIF.index:   ## i is each OTL (rows in LOIF matrix)
    A = LOIF.loc[i,:]   #A becomes a pandas series where each element corresponds to an outage in the system
    Ta_sets[i] = A     #This is the initialization stage where we begin with all outages possible outages for each observation point.
    # print(Ta_sets[i])

Ta_sets = pd.DataFrame(Ta_sets)


# #This function create a KNN model where we specify the which features we are interested in and the outages we want to test using k neighbors


def knn_model(ML_labels,OL_labels,k):
    #In training data we filter out data to only look at the obsevation points (Features) and outages we are interested in.
    Training_BD = Training_all.loc[OL_labels,ML_labels]

    # Here we use MinMaxscaler to convert our measurments into values from 0 to 1
    X = Training_BD.iloc[:,:-1]  ## Features
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
    # print(X)
    y = Training_BD.iloc[:,-1]  ## Labels 

    #From the same training data, we split it into training and testing data to run KNN
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

    #Set KNN classifier, fit our model and make predictions on testing labels
    knn = KNeighborsClassifier(n_neighbors=k,weights='uniform',algorithm='auto',p=3)
    #Train KNN with training data from split
    knn.fit(x_train, y_train)
            #X_train, y_train
    Predictions = knn.predict(x_test)
    True_Labels = y_test

    #Collect classification report between predicted labels and real labels from testing data
    Report= classification_report(True_Labels,Predictions,output_dict=True,zero_division=0)
    # print(classification_report(True_Labels,Predictions))
    return Report


#Calculate Minimum Gamma
def calc_min_gamma_beta(k,desired_precision,desired_f1score):

    #Dataframe Lists
    final_sets = []   #Will append every Sa subset after finding minimum values
    set_lenghts = []  #Will append the size of every final Sa subset of each observed transmission line
    min_gammas = []   #Will append the minimum gamma for each observed transmisson line
    min_betas  = []   #Will append the minimum beta for each observed transmission line
    final_f1s = []
    index_col = []


    for i in LOIF.index:
        print(f'Line {i}')
        Ta = Ta_sets[i]
        Sa = pd.Series(Ta)
        gamma = 1e-30
        beta = 1e-30
        ML_labels = []
        ML_labels.append(f'PF Line {i}')  
        ML_labels.append(f'QF Line {i}')
        ML_labels.append(f'PT Line {i}')
        ML_labels.append(f'QT Line {i}')
        ML_labels.append(f'Label')

        current_precision = 0.0
        while(current_precision < desired_precision):
            Sa = Sa[abs(Sa) >= gamma]       #Find LOIFs greater than gamma
            OL_labels = [0] + Sa.index.to_list()   
            
            Report = knn_model(ML_labels=ML_labels,OL_labels=OL_labels,k=8)
            current_precision = round(Report['0']['precision'], 2)


            if current_precision >= desired_precision:
                mingamma = gamma
            elif gamma < 0.1:
                gamma = gamma*(1e1)
            else:
                gamma = round(gamma + 0.1,2)

        current_f1score = 0.0
        while(current_f1score < desired_f1score):
            Sa = Sa[abs(Sa) >= mingamma]
            Sa = Sa.sort_values(ascending=False)  #Organize in descending order

            keep_LOIFs = [float(Sa.iloc[0])]
            keep_Index = [int(Sa.index[0])]

            for j in Sa.index[1:]:
                if (abs(keep_LOIFs[-1]-Sa.loc[j])) >= beta:
                    keep_LOIFs.append(float(Sa.loc[j]))
                    keep_Index.append(int(j))

            Sa = Sa[keep_Index]
            OL_labels = [0] + Sa.index.to_list()
            Report = knn_model(ML_labels=ML_labels,OL_labels=OL_labels,k=k)
            current_f1score = Report['macro avg']['f1-score']       #Get F1-Score from classification report
            if (current_f1score >= desired_f1score) or (len(keep_Index)==1):
                minbeta = beta
                final_f1s.append(current_f1score)
                break
            elif beta < 0.1:
                beta = beta*(1e1)
            else:
                beta = round(beta + 0.1,2) 
     

        min_gammas.append(mingamma)
        min_betas.append(minbeta) 
        set_lenghts.append(len(Sa.index.to_list()))    
        final_sets.append(Sa.index.to_list())
        index_col.append(i)

    DF = pd.DataFrame({'Lines':index_col,'min_gamma':min_gammas,'min_beta':min_betas,'Total_outages':set_lenghts,'Sa_Subsets':final_sets,'Actual_F1-Score':final_f1s})
    DF=DF.set_index('Lines')

        
    DF.to_csv(f'Minimum_gb_k{k}_dp{desired_precision}_df{desired_f1score}')
    return DF


def max_coverage_greedy(mgb_df,OTL_num):
    OTLs = []
    print(mgb_df)
    print('-----------------GREEDY ALGORITHM-------------------')
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
        print(f'Largest Subset (Iteration {i+1}): {largest_OPL} --> {diffs[largest_OPL]}')
        if diffs[largest_OPL] == 0:  #If there are no more new elements stop iteration
            break
        else:                        #If there are still uncovered elements continue iteration and update collection set.
            OTLs.append(int(largest_OPL))
            current_set = mgb_df.loc[int(largest_OPL),'Sa_Subsets']
            current_set = ast.literal_eval(current_set)
            collection_set = list(set(collection_set + current_set))

    print(f'OTLs ({len(OTLs)}):\n',OTLs)

    print(f'Covered Outages({len(collection_set)}):\n',collection_set)
    ML_labels = []
    for i in OTLs:
        ML_labels.append(f'PF Line {i}')  
        ML_labels.append(f'QF Line {i}')
        ML_labels.append(f'PT Line {i}')
        ML_labels.append(f'QT Line {i}')
    ML_labels.append(f'Label')

    OL_labels = [0] + collection_set

    Report = knn_model(k=8,ML_labels=ML_labels,OL_labels=OL_labels)
    print(pd.DataFrame(Report))

def main(k,desired_precision,desired_f1):
    print('================  k-neighbors={0}, Desired Precision(For Normal Conditions)={1}, Desired F1-Score (Overall Performance)={2} ================'.format(k,desired_precision,desired_f1))
    #Checks if minimum gamma results already exist, if not it calculates them first
    os.chdir(folder_path)                     #Update path to new folder (Min Gamma Results)
    if os.path.exists(f'Minimum_gamma_per_OPL_k{k}_dp{desired_precision}_df1{desired_f1}.csv'):     #If results for a certain precision and F1-score exist read the CSV file contaiing the reults
        mgb_df = pd.read_csv(f'Minimum_gamma_per_OPL_k{k}_dp{desired_precision}_df1{desired_f1}.csv',index_col=0)  #Read CSV file
    else:
        os.chdir(current_path)                     #Update path to new folder (Min Gamma Results)                                                                                     #IF no results exists, begin calculating minimum gamma and save results to a CSV file.
        mgb_df = calc_min_gamma_beta(k=8,desired_f1score=desired_precision,desired_precision=desired_f1)
        os.chdir(folder_path)                     #Update path to new folder (Min Gamma Results)
        mgb_df.to_csv(f'Minimum_gamma_per_OPL_k{k}_dp{desired_precision}_df1{desired_f1}.csv')
        mgb_df = pd.read_csv(f'Minimum_gamma_per_OPL_k{k}_dp{desired_precision}_df1{desired_f1}.csv',index_col=0)  #Read CSV file
        
    
    max_coverage_greedy(mgb_df=mgb_df.sort_values(by='Total_outages',ascending=False),OTL_num=1000)

for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    for j in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        main(k=8,desired_f1=i,desired_precision=j)


print('===================Code Executed============================')
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Code executed in {elapsed_time:0.4f} Seconds")
elapsed_time = elapsed_time/60
print(f"Code executed in {elapsed_time:0.4f} Minutes")
elapsed_time = elapsed_time/60
print(f"Code executed in {elapsed_time:0.4f} Hours")
