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

def select_files(t):
    if t == 1:
        title = "SELECT CONVERGENCE FILE OF THE SYSTEM YOU ARE INTERESTED IN"
    elif t == 2:
        title = "SELECT BRANCH DATA FILE OF THE SYSTEM YOU ARE INTERESTED IN (MAKE SURE IT IS THE SAME SYSTEM YOU INITIALLY CHOSE)"
    elif t == 3:
        title = 'SELECT LOIF/LODF MATRIX FILE OF THE SYSTEM YOU ARE INTERESTED IN (MAKE SURE IT IS THE SAME SYSTEM YOU INITIALLY CHOSE)'
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
    knn = KNeighborsClassifier(n_neighbors=k,weights='uniform',algorithm='auto',p=3)
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
# 
# In our previous work we wanted to see the effect of different gamma and beta values and we varied each variable by [0.1, 0.2, 0.4].
#  With Sa we created a observability metric, eta, which we used to select a set of OTLs of sizes [2, 4, 8]. 
#  Because eta is directly influenced by gamma and beta, the OTLs that are selected might change which can affect ML performance. 
# In order to find the best set of detectable outages, we would like to find the optimal values of gamma and beta for each OTL that will result in good ML performance.
#
# There are two classification measures that we will use to find the best gamma and beta:
#   - Precision: True Positives / (True Positves + False Positives)
#       True Positives: The number samples of a particular class (outage) that the ML model predicted correctly
#       False Positives: The number of samples that were incorrectly predicted as the class (outage) by the ML model.
#      **IF Precision is 1.0, then we know that there are no False Positives meaning that all the samples of the particular class were predicted correctly.
#   -Avg. F1-Score: The harmonic mean of precision and recall, in other words, overall performance metric.
#      **IF Avg. F1-Score is 1.0, then we know that we got 100% accuracy on classification of all labels (outages).
# 
# In order to find the minimum/optimal gamma, we use precision specifically for the class representing normal conditions (0). 
#    - If precision for normal conditions is 1.0, then we know that the outages in the detectable outage set (Sa) provides a significant change in power flow on the OTL 
#     (enough not to be confused with the power flow during normal conditions)
#    - This is done through an iterative process, where we first start with a very low gamma and increase by small increments with each iteration. 
#      In each iteration we find Sa using the current gamma and run KNN classifier as our ML model to collect a classification report.
#      From this report, we observe the precision for normal conditions and check if it meets desired value of 1.0 (desired precision)
#      Once we reach desired precision, we found our optimal/minimum gamma for OTL. 
#
# In order to find the minimum/optimal best, we use Avg. F1-Score (overall peformance) for all classes.
#    -If Avg. F1-Score is 1.0, then we know that the outages in Sa have LOIFs that are separated from each other by a minimum distance of beta and all outages are predicted correctly.
#    -This is done through an iterative process, where we first start with a very low beta and increase by small increments with each iteration.
#     In each iteration, we find Sa using the current beta and run KNN classifier as our ML model to collect a classification report.
#     From this report, we observe the Avg. F1-Score and check if it meets desired value of 1.0 (desire f1-score)
#     Once we reach desired f1-score, we found our optimal/minimum beta for OTL.
def lod_detectable_subsetgen(k,desired_precision,desired_f1score,f,LOIF,Ta_sets,Training_all):
    print('Finding Sa Subsets......')
    #Dataframe Lists
    final_sets = []   #Will append every Sa subset after finding minimum values
    set_lenghts = []  #Will append the size of every final Sa subset of each observed transmission line
    min_gammas = []   #Will append the minimum gamma for each observed transmisson line
    min_betas  = []   #Will append the minimum beta for each observed transmission line
    final_f1s = []    #Will append the avg. f1 scores for each observed transmsission line
    index_col = []    #Will append the line number for each observed transmission line


    for i in LOIF.index[:]:
        Ta = Ta_sets[i]      #Ta will hold the LOIFs of all outages for current OTL i (grab from dict.)   #create pandas serires Sa (initially holds LOIFs of all outages)

        gamma = 1e-30       #start with low gamma
        beta = 1e-30        #start with low beta

        #pull columns from Training Data that are needed (Power Flows of OTL + Label)
        #stops from having to access whole dataset.
        otl_col = []        #column labels of OTL to filter Training Data 
        otl_col.append(f'PF Line {i}')  #Active Power at From Bus
        otl_col.append(f'QF Line {i}')  #Reactive Power at From Bus
        otl_col.append(f'PT Line {i}')  #Active Power at To Bus
        otl_col.append(f'QT Line {i}')  #Reactive Power at To Bus
        otl_col.append(f'Label')        #Class labels

        #Find Min. Gamma
        current_precision = 0.0  #intialize current precision with zero (will update with each iteration)
        while(1): #Keep looping until minimum gamma is found
            Sa = pd.Series(Ta)              #create pandas serires Sa (initially holds LOIFs of all outages)
            Sa = Sa[abs(Sa) >= gamma]       #Find LOIFs greater than gamma
    
            #Get the line numbers of the outages that satisfied current gamma condition.
            outage_labels = [i] + Sa.index.to_list()   #Always include the line number of the OTL (Easy to detect, since no power should be flowing through OTL)
            outage_labels = list(set(outage_labels))   #used to pull the rows from Training Data for the following classes

            #Runn KNN classifier (Make Sure to include normal condtions, 0, in outage_labels)
            Report = lod_exp_exec(otl_col=otl_col,outage_labels=[0] + outage_labels,k=k,output=False,f=f,Training_all=Training_all)
            current_precision = round(Report['0']['precision'], 2)    #Collect Precision For Normal Conditions (Label/Class: 0)

            #Check if current precision meets the desired precision of 1.0 (or other value depending on user)
            #If desired precision is reached OR Sa subset if of OTL hase length of 1, or if gamma passes the value 10 (Loop Stopper, usually means it failed to reach desired precision)
            if (current_precision >= desired_precision) or (len(Sa)==1) or (gamma == 10):  #If desired precision is reached save min gamma and print classification report to Txt file
                mingamma = gamma
                print(f'Line {i} --> gamma: {mingamma}',file=f)
                Report = lod_exp_exec(otl_col=otl_col,outage_labels=[0]+outage_labels,k=k,output=True,f=f,Training_all=Training_all)
                break
            elif gamma < 0.1:   #If desired precision is not met, and gamma is below 0.1, increase by a factor of 1e1
                gamma = gamma*(1e1)
            else:              #If desired precision is not met, and gamma is above 0.1, increase in increments of 0.01
                gamma = round(gamma + 0.01,2)

        #Find Min. Beta
        current_f1score = 0.0  #initalize current f1score with zero (will update with each iteration)
        while(1): #Keep looping until minimum beta is found
            Sa = pd.Series(Ta)  #create pandas serires Sa (initially holds LOIFs of all outages)
            Sa = Sa[abs(Sa) >= mingamma]  #Include min. gamma threshold
            Sa = Sa.sort_values(ascending=False)  #Organize LOIFs in descending order
            
            #Find LOIFs whose abs. differences are greater than beta, provides LOIFs that are separated by a distance of beta
            Sa = Sa[Sa.diff().abs().fillna(beta) >= beta]  

            outage_labels = [i] + Sa.index.to_list()  #Always include the line number of the OTL (Easy to detect, since no power should be flowing through OTL)
            outage_labels = list(set(outage_labels))  #used to pull the rows from Training Data for the following classes
            
            #Runn KNN classifier (Make Sure to include normal condtions, 0, in outage_labels)
            Report = lod_exp_exec(otl_col=otl_col,outage_labels=[0] + outage_labels,k=k,output=False,f=f,Training_all=Training_all)
            current_f1score = Report['macro avg']['f1-score']       #Collect Avg. F1-Score (Overall Performance)
            
            #Check if current f1-score meets the desired f1-score of 1.0 (or other value depending on user)
            #If desired f1-score is reached OR length of Sa is 1 OR if beta passes the value 10 (Loop Stopper, usually means it failed to reach desired F1-Score)
            if (current_f1score >= desired_f1score) or (len(Sa)==1) or (beta == 10): #If desired f1-score is reached save min beta and print classification report to Txt file
                minbeta = beta
                print(f'Line {i} --> beta: {minbeta}',file=f)
                Report = lod_exp_exec(otl_col=otl_col,outage_labels=[0] + outage_labels,k=k,output=True,f=f,Training_all=Training_all)
                final_f1s.append(current_f1score)
                break
            elif beta < 0.1:   #If desired f1-score is not met, and beta is below 0.1, increase by a factor of 1e1
                beta = beta*(1e1)
            else:              #If desired f1-score is not met, and gamma is above 0.1, increase in increments of 0.01
                beta = round(beta + 0.01,2) 
     
        min_gammas.append(mingamma)
        min_betas.append(minbeta) 
        set_lenghts.append(len(outage_labels))    
        final_sets.append(outage_labels)
        index_col.append(i)

    #Save Reults for Each OTL into DataFrame and Return DataFrame
    DF = pd.DataFrame({'Lines':index_col,'min_gamma':min_gammas,'min_beta':min_betas,'Total_outages':set_lenghts,'Sa_Subsets':final_sets,'Actual_F1-Score':final_f1s})
    DF=DF.set_index('Lines')
    return DF



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

def lod_otl_select(mgb_df,f,Training_all,LOIF):
    Results = {}
    print(f'Final Sa Subsets:\n {mgb_df}',file=f)
    print('\n',file=f)
    Results['MCP'] = max_coverage_greedy(mgb_df=mgb_df.sort_values(by='Total_outages',ascending=False),OTL_num=1000,f=f,Training_all=Training_all)
    print('\n',file=f)
    Results['Eta'] = high_eta(mgb_df=mgb_df.sort_values(by='Total_outages',ascending=False),MCP= Results['MCP'],f=f,Training_all=Training_all)
    print('\n',file=f)
    Results['Random'] = random_OTL(MCP=Results['MCP'],f=f,Training_all=Training_all,LOIF=LOIF)




##MAIN
def main(k,desired_precision,desired_f1):
    start_time = time.perf_counter()
    root = tk.Tk()
    root.withdraw() 

    #Select Which Files to Use
    fpath1 = select_files(1)
    Convergence_Data = pd.read_csv(fpath1,index_col=0,header=None)
    fpath2 = select_files(3)
    LOIF = pd.read_csv(fpath2,index_col=-1)  #Excluded lines are not removed in matlab
    if 'LODF' in fpath2:
        case = re.search(r"LODFmatrix_(case.+)\.csv", fpath2).group(1)
        matrix = 'LODF'
    elif 'LOIF' in fpath2:
        case = re.search(r"LOIFmatrix_(case.+)\.csv", fpath2).group(1)
        matrix = 'LOIF'
    
    fpath3 = select_files(2)
    Training_all = pd.read_csv(fpath3)
    Training_all.index = Training_all['Label']

    sol_type = fpath3[fpath3.index('.csv')-2:fpath3.index('.csv')]
    print(sol_type)
    data_label = re.search(r'([^/]+)_',fpath3).group(1)
    data_label = data_label[:data_label.index('_')]


    current_path = os.getcwd()   #Get the address of current path
    folder_path = f'{current_path}/Final_Sa_{case}_{data_label}_{sol_type}_{matrix}' #Create Path for New Folder to store (Min Gamma Results)
    os.makedirs(folder_path, exist_ok=True)   #If the Folder already exist do nothing, if not then create new folder

    os.chdir(folder_path)
    f = open(f"output_{case}_{data_label}_{sol_type}_{matrix}.txt", 'w',)
    print(case,file=f)
    print(f'DC or AC: {sol_type}',file=f)
    print(data_label,file=f)


    #Script Calculates How Many Transmission Lines There are
    num_lines = len(LOIF.index.to_list())
    print(f'Total Number of Transmission Lines: ',num_lines,file=f)
    #Because We Ignore outages that don't converge when disconnected, we want to see all outages that we collected data on (possible outages)
    #Script Tells us how many lines converged, which are the lines we will focus on.
    outage_set = Convergence_Data.loc[Convergence_Data[1] ==1].index.to_list()
    outage_set = outage_set[1:]
    #Script Tells us how many lines did not converge, which are the lines we will ignore
    excluded_set = Convergence_Data.loc[Convergence_Data[1] ==0].index.to_list()
    print(f'Possible Line Outages ({len(outage_set)}):\n',outage_set,file=f)
    print(f'Excluded Transmission Lines ({len(excluded_set)}):\n',excluded_set,file=f)


    #Add row labels to LOIF matrix
    line_numbers = [i for i in range(1,num_lines+1)]
    LOIF.index = line_numbers
    LOIF.columns = line_numbers
    LOIF = LOIF.drop(index=excluded_set,columns=excluded_set)  #Remove Rows and Columns related to excluded lines

    Ta_sets = {}
    for i in LOIF.index:   ## i is each OTL (rows in LOIF matrix)
        A = LOIF.loc[i,:]   #A becomes a pandas series where each element corresponds to an outage in the system
        Ta_sets[i] = A     #This is the initialization stage where we begin with all outages possible outages for each observation point.
        # print(Ta_sets[i])

    Ta_sets = pd.DataFrame(Ta_sets)
    
    print('================  k-neighbors={0}, Desired Precision(For Normal Conditions)={1}, Desired F1-Score (Overall Performance)={2} ================'.format(k,desired_precision,desired_f1),file=f)
    #Checks if minimum gamma results already exist, if not it calculates them first
    os.chdir(folder_path)                     #Update path to new folder (Min Gamma Results)
    if os.path.exists(f'Minimum_gb_k{k}_dp{desired_precision}_df{desired_f1}_{sol_type}.csv'):     #If results for a certain precision and F1-score exist read the CSV file contaiing the reults
        mgb_df = pd.read_csv(f'Minimum_gb_k{k}_dp{desired_precision}_df{desired_f1}_{sol_type}.csv',index_col=0)  #Read CSV file
    else:
        mgb_df = lod_detectable_subsetgen(k=8,desired_f1score=desired_f1,desired_precision=desired_precision,LOIF=LOIF,Ta_sets=Ta_sets,f=f,Training_all=Training_all)
        mgb_df.to_csv(f'Minimum_gb_k{k}_dp{desired_precision}_df{desired_f1}_{sol_type}.csv')
        mgb_df = pd.read_csv(f'Minimum_gb_k{k}_dp{desired_precision}_df{desired_f1}_{sol_type}.csv',index_col=0)  #Read CSV file
    

    lod_otl_select(mgb_df=mgb_df,f=f,Training_all=Training_all,LOIF=LOIF)
    print('===================Code Executed============================',file=f)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Code executed in {elapsed_time:0.4f} Seconds",file=f)
    elapsed_time = elapsed_time/60
    print(f"Code executed in {elapsed_time:0.4f} Minutes",file=f)
    elapsed_time = elapsed_time/60
    print(f"Code executed in {elapsed_time:0.4f} Hours",file=f)


    print(fpath1,file=f)
    print(fpath2,file=f)
    print(fpath3,file=f)
    f.close()

main(k=8,desired_f1=1.0,desired_precision=1.0)
