import pandas as pd
import random
import os

############################ Find Dectectable subsets (lod_detectable_subsetgen)============================================
# This function uses "beta", "gamma", and the LOIF/LODF to find detectable Sa subsets for each OTL in the system.
# Beta filters values that are small, removing outages that have little to no impact
# Gamma filters values that are not separated by a minimum distance, removing outages that have similar impact. 
# Parameters
# beta: Used to filter LOIF/LODF values that are small to point where they are not detectable (outages with little to no change in power flow at the OTL)
# gamma: Used to filter LOIF/LODF values that are not separated by a minimum distance of gamma (outages with similar change in power flow at the OTL)
# system: This is the name of the case system the use is interested in. Ex. 'case_ieee30','case118','case_ACTIVSg2000','caseWisconsin_1664'
# data_dir: This is the folder path containing you important files needed to run to code such as the
#            LOIF/LODF matrix, convergence data, and training data
# matrix: The user has the option to use LOIF or LODF matrices
# Output
# Sa_subsets: a list of lists containing the detectable susbets (Sa) of all OTLs
def lod_detectable_subsetgen(beta,gamma,system,data_dir,matrix):
    print('Finding Sa Subsets......')
    folder_path = data_dir
    os.chdir(folder_path)
    matrix_file = f"{matrix}matrix_{system}.csv"
    LOIF = pd.read_csv(matrix_file,index_col=-1)  #Excluded lines are not removed in matlab
    num_lines = len(LOIF.index)
    line_numbers = range(1,num_lines+1)
    LOIF.index = line_numbers
    LOIF.columns = line_numbers

    excluded_set = LOIF.columns[(LOIF == 0).all()].to_list()

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

## Check function for Self Containment
# Sa=lod_detectable_subsetgen(beta=0.1,gamma=0.1,system='case118',data_dir=r"C:\Users\dflores\Documents\Python\Total_coverage_tests\case118",matrix='LOIF')
# print(Sa)

## OTL Selection: Max Coverage Problem
# Using the results we obtained from "lod_detectable_subsetgen" (in "df"). We perform an iterative process where we select subsets
# that offer the most uncoverged outages in each iteration. These outages are stored in a collection set (initially empty). This process is repeated
# until we have selected a specific number of OTLs (based on "OTL_num") or until we have covered all possible outages (Full Coverage).
# Parameters
# Sa: This is the list of lists containing our detectable subsets from "lod_detectable_subsetgen"
# X: The number of OTLs to select in MCP algorithm (1,2,4,8, or Full Coverage)
# Output
# OTLs: This is the set of OTLs that were selected using MCP

###############Psuedo code from Journal########################################
# 1: Input: Set of all lines: L, collection of sa sets: S, desired number of sets: X
# 2: Output: Selected sets: C, coverage size: |C|
# 3: C ←∅
# 4: R ←∅
# 5: U ←L
# 6:
# 7: for i = 1 to k do
# 8: if U = ∅ then
# 9: break
# 10: end if
# 11: S∗ ←argmaxS∈S\C |S ∩U|
# 12: C ←C∪{S∗}
# 13: R←R∪S∗
# 14: U ←U\S∗
# 15: end for
# 16: Return (C,|C|)
def mcp(Sa,X):
    Sa_lengths = [len(i) for i in Sa]
    dict = {}
    dict['Sa_Subsets'] = Sa
    dict['Total_Outages'] = Sa_lengths
    df = pd.DataFrame(dict,index=range(1,len(Sa)+1))

    if X == 'FC':
        X = 30000
    print('Starting MCP Algorithm......')
    C = []  # Currently Selected sets
    R = []  # Set of Covered Outages

    # Initialize U as the set of all outages (uncovered)
    num_lines = len(Sa)
    U = list(range(1, num_lines + 1))

    for i in range(0,X):
        diffs = {}  # Will hold the intersection size with U for each subset

        # This for loop will go through Sa subsets to get the intersection with U
        for j in df.index:  # For every observation point
            current_set = df.loc[j,'Sa_Subsets']

            # Calculate the intersection with uncovered set U
            diffs[f'{j}'] = len(list(set(current_set).intersection(U)))

        #S∗ ←argmaxS∈S\C |S ∩ U|
        largest_OTL = max(diffs, key=diffs.get)   # Gets the Sa subset with the largest intersection with U

        if diffs[largest_OTL] == 0:  # If U is empty or no intersection, stop iteration
            break
        else:                        # If there are still uncovered outages, continue iteration and update U
            
            C.append(int(largest_OTL))  #Update list of Selected Sets with current OTL selected
            R = list(set(R + [int(largest_OTL)])) #Update the list of covered outages with outages covered by selected OTL

            current_set = df.loc[int(largest_OTL),'Sa_Subsets']

            #Update set of uncovered outages (remove outages that are covered with selected OTL)
            U = list(set(U) - set(current_set))

    return C  #Only return the set of selected OTLs, we will predict all outages using this set. We can get |C| using len(C)

## Check function for Self Containment
# MCP_otls = mcp(Sa=Sa,X='FC')   #'FC' for full coverage
# print(MCP_otls)


## OTL Selection: High Eta
# This function uses the results in MCP to determine the number of OTLs to select using High Eta Lines.
# From the list of lists containing all Sa subsets, we use a dataframe to sort the subsets in descending order based on their sizes.
# The subsets at the top represent High Eta Lines.
# Parameters
# Sa: This is the list of lists containing our detectable subsets from "lod_detectable_subsetgen"
# X: The number of OTLs to select in MCP algorithm (1,2,4,8, or Full Coverage)
#  Output
#  Eta_OTLs: Set of OTLs that were selected using High Eta lines.
def high_eta(Sa,X):
    Sa_lengths = [len(i) for i in Sa]
    dict = {}
    dict['Sa_Subsets'] = Sa
    dict['Total_Outages'] = Sa_lengths
    df = pd.DataFrame(dict,index=range(1,len(Sa)+1))

    print('Starting High Eta Tests.......')
    # print('------------------HIGH ETA Test-----------------------------------------')
    df = df.sort_values('Total_Outages',ascending = False)
    Eta_OTLs = []
    for i in df.index[:X]:
        Eta_OTLs.append(i)
    # print(f'High Eta Lines ({len(Eta_OTLs)}): {Eta_OTLs}')

    # print('-----------------End of HIGH ETA Test---------------------------------')
    return Eta_OTLs

## Check function for Self Containment
# HE = high_eta(Sa=Sa,X=34)
# print(HE)



## OTL Selection: Random
# This function uses the results in MCP to determine the number of OTLs to select by random in the overall set of OTLs.
# We collect 10 different set of random OTLs that will be used to get the average performance in F1-Score,Recall, or Precision
#  Parameters
# Sa: This is the list of lists containing our detectable subsets from "lod_detectable_subsetgen". Use only to determine the total number of lines
# X: The number of OTLs to select in MCP algorithm (1,2,4,8, or Full Coverage)
#  Output
#  rand_list: list of list containing the sets of OTLs that were selected by random (Should be 10 sets of random OTLs)
def random_otl(Sa,X):
    print('Starting Random OTL Tests......')
    num_lines = len(Sa)
    line_numbers = range(1,num_lines+1) 
    # print('------------------RANDOM OTL SELECTION------------------------------------')
    rand_otls = random.sample(line_numbers, X,)
    # print('-------------------END of RANDOM OTL SELECTION---------------------------')
    return rand_otls

## Check function for Self Containment
# Rand = random_otl(Sa=Sa,X=34)
# print(Rand)