import pandas as pd
import random


## OTL Selection: Max Coverage Problem
# Using the results we obtained from "lod_detectable_subsetgen" (in "df"). We perform an iterative process where we select subsets
# that offer the most uncoverged outages in each iteration. These outages are stored in a collection set (initially empty). This process is repeated
# until we have selected a specific number of OTLs (based on "OTL_num") or until we have covered all possible outages (Full Coverage).
# Parameters
# df: This is the dataframe of our detectable subsets from "lod_detectable_subsetgen" inside the main script lod_experiments.py
# OTL_num: The number of OTLs to select in MCP algorithm (1,2,4,8, or Full Coverage)
# Output
# OTLs: This is the set of OTLs that were selected using MCP
def mcp(df,OTL_num):
    if OTL_num == 'FC':
        OTL_num = 30000
    print('Starting MCP Algorithm......')
    OTLs = []

    # print('-----------------MCP GREEDY ALGORITHM-------------------')
    collection_set = []  #Will be used to store the outages we have already covered
    for i in range(0,OTL_num):
        diffs = {} #Will hold the number of new elements each subset has so that we can compare all of them at the end
        

        #This for loop will go through Sa subsets to get the difference between each subset and and the collection set for current iteration
        for j in df.index:  #For every observation point
            current_set = df.loc[j,'Sa_Subsets']

            #Calculate the diffrence between current iteration sets and collection set to see which outages are not covered yet
            diffs[f'{j}'] = len(list(set(current_set).difference(collection_set)))   #Counts the number of uncovered outages

        largest_OPL = max(diffs, key=diffs.get)   #Gets the Sa subset with the largest number of elements not yet covered in collection set.
        # print(f'Largest Subset (Iteration {i+1}): {largest_OPL} --> {diffs[largest_OPL]}')
        if diffs[largest_OPL] == 0:  #If there are no more uncovered outages stop iteration
            break
        else:                        #If there are still uncovered outages, continue iteration and update collection set.
            OTLs.append(int(largest_OPL))
            current_set = df.loc[int(largest_OPL),'Sa_Subsets']
            collection_set = list(set(collection_set + current_set))

    # print(f'OTLs ({len(OTLs)}):\n',OTLs)

    # print(f'Covered Outages({len(collection_set)}):\n',collection_set)
    # print('---------------------End Of MCP GREEDY ALGORITHM----------------------')

    return OTLs

## OTL Selection: High Eta
# This function uses the results in MCP to determine the number of OTLs to select using High Eta Lines.
# From the list of lists containing all Sa subsets, we use a dataframe to sort the subsets in descending order based on their sizes.
# The subsets at the top represent High Eta Lines.
# Parameters
#  MCP: This is the results of the MCP OTL selection, (Mainly Used to find the number of OTLs selected)
#       This is needed in the case of 'Full Coverage', which is determined by MCP results
#  df: This is the dataframe of our detectable subsets from "lod_detectable_subsetgen" inside the main script lod_experiments.py
#  Output
#  Eta_OTLs: Set of OTLs that were selected using High Eta lines.
def high_eta(MCP,df):
    print('Starting High Eta Tests.......')
    # print('------------------HIGH ETA Test-----------------------------------------')
    df = df.sort_values('Total_Outages',ascending = False)
    Eta_OTLs = []
    for i in df.index[:len(MCP)]:
        Eta_OTLs.append(i)
    # print(f'High Eta Lines ({len(Eta_OTLs)}): {Eta_OTLs}')

    # print('-----------------End of HIGH ETA Test---------------------------------')
    return Eta_OTLs

## OTL Selection: Random
# This function uses the results in MCP to determine the number of OTLs to select by random in the overall set of OTLs.
# We collect 10 different set of random OTLs that will be used to get the average performance in F1-Score,Recall, or Precision
#  Parameters
#  MCP: This is the results of the MCP OTL selection, (Mainly Used to find the number of OTLs selected)
#       This is needed in the case of 'Full Coverage', which is determined by MCP results
#  LOIF: LOIF matrix which is mainly used to obtain the set of all OTLs in order to set a random set of OTLs.
#  Output
#  rand_list: list of list containing the sets of OTLs that were selected by random (Should be 10 sets of random OTLs)
def random_otl(MCP,LOIF):
    print('Starting Random OTL Tests......')
    # print('------------------RANDOM OTL SELECTION------------------------------------')
    rand_list = []
    for i in range(1,11):
        # print(f'Random Test {i}')
        rand_OTLs = random.sample(LOIF.index.to_list(), len(MCP),)
        # print(f'Random OTLs ({len(rand_OTLs)}): {rand_OTLs}\n')
        rand_list.append(rand_OTLs)
    # print('-------------------END of RANDOM OTL SELECTION---------------------------')
    return rand_list