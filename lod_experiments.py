
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


def lod_detectable_subsetgen(k,desired_precision,desired_f1score,f,LOIF,Ta_sets,Training_all):
    print('Finding Sa Subsets......')
    #Dataframe Lists
    final_sets = []   #Will append every Sa subset after finding minimum values
    set_lenghts = []  #Will append the size of every final Sa subset of each observed transmission line
    min_gammas = []   #Will append the minimum gamma for each observed transmisson line
    min_betas  = []   #Will append the minimum beta for each observed transmission line
    final_f1s = []
    index_col = []


    for i in LOIF.index[:]:
        # print(f'Line {i}')
        # print(f'Line {i}')
        Ta = Ta_sets[i]
        Sa = pd.Series(Ta)
        # print(Sa)
        gamma = 1e-30
        beta = 1e-30
        otl_col = []
        otl_col.append(f'PF Line {i}')  
        otl_col.append(f'QF Line {i}')
        otl_col.append(f'PT Line {i}')
        otl_col.append(f'QT Line {i}')
        otl_col.append(f'Label')

        current_precision = 0.0
        while(1):
            Sa = pd.Series(Ta)
            Sa = Sa[abs(Sa) >= gamma]       #Find LOIFs greater than gamma
            # print(Sa)
            outage_labels = [i] + Sa.index.to_list()   
            outage_labels = list(set(outage_labels))
            Report = lod_exp_exec(otl_col=otl_col,outage_labels=[0] + outage_labels,k=k,output=False,f=f,Training_all=Training_all)
            current_precision = round(Report['0']['precision'], 2)


            if (current_precision >= desired_precision) or (len(Sa)==1) or (gamma == 10):
                mingamma = gamma
                print(f'Line {i} --> gamma: {mingamma}',file=f)
                Report = lod_exp_exec(otl_col=otl_col,outage_labels=outage_labels,k=k,output=True,f=f,Training_all=Training_all)
                break
            elif gamma < 0.1:
                gamma = gamma*(1e1)
            else:
                gamma = round(gamma + 0.01,2)

        current_f1score = 0.0
        while(1):
            Sa = pd.Series(Ta)
            Sa = Sa[abs(Sa) >= mingamma]
            Sa = Sa.sort_values(ascending=False)  #Organize in descending order
            
            Sa = Sa[Sa.diff().abs().fillna(beta) >= beta]

            outage_labels = [i] + Sa.index.to_list()
            outage_labels = list(set(outage_labels))
            Report = lod_exp_exec(otl_col=otl_col,outage_labels=[0] + outage_labels,k=k,output=False,f=f,Training_all=Training_all)
            current_f1score = Report['macro avg']['f1-score']       #Get F1-Score from classification report
            if (current_f1score >= desired_f1score) or (len(Sa)==1) or (beta == 10):
                minbeta = beta
                print(f'Line {i} --> beta: {minbeta}',file=f)
                Report = lod_exp_exec(otl_col=otl_col,outage_labels=outage_labels,k=k,output=True,f=f,Training_all=Training_all)
                final_f1s.append(current_f1score)
                break
            elif beta < 0.1:
                beta = beta*(1e1)
            else:
                beta = round(beta + 0.01,2) 
     
        min_gammas.append(mingamma)
        min_betas.append(minbeta) 
        set_lenghts.append(len(outage_labels))    
        final_sets.append(outage_labels)
        index_col.append(i)

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
    data_label = re.search(rf"Branchdata_{case}_(.+?)_{sol_type}\.csv",fpath3).group(1)

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
