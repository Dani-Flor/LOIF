import pandas as pd
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

####################################### KNN classifier #########################################################
# This function filters our training data by focusing only on the OTLs we are interested for all outage scenarios plus normal conditions.
# The we use MinMaxScaler to convert our power flow measurements into values between 0 and 1. After that we then use train_test_split to use a portion of
# our training data as testing data.
# After that we then fit our training data to a KNN classifier and predict the labels of our testing data.
# Parameters
# - otl_set: The columns in training data representing the OTLs that are selected in "lod_otl_select.py" for MCP, High Eta, and Random
# - k: Number of Neighbors to consider in KNN
# - output: TRUE/FALSE, if true prints classification report to screen
# - training_data: This is the labeled power flow data obtained from MATLAB script "lod_labeled_datagen.m"
# Output
# Report: The classification report (in dictionary form)
def lod_exp_exec(otl_set,k,output,training_data):
    # In training data we filter out data to only look at the selected OTLs (Features)
    column_labels = []
    for j in otl_set:
        column_labels.append(f'PF Line {j}')  
        column_labels.append(f'QF Line {j}')
        column_labels.append(f'PT Line {j}')
        column_labels.append(f'QT Line {j}')
    column_labels.append('Label')


    Training_BD = training_data.loc[:,column_labels]

    # Here we use MinMaxscaler to convert our measurements into values from 0 to 1
    X = Training_BD.iloc[:,:-1]  ## Features
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)
    # print(X)
    y = Training_BD.iloc[:,-1]  ## Labels 

    # From the same training data, we split it into training and testing data to run KNN
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

    # Set KNN classifier, fit our model and make predictions on testing labels
    knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='kd_tree', p=2)  #p=2 is euclidean distance, p=1 is manhattan distance
    # Train KNN with training data from split
    knn.fit(x_train, y_train)
            #X_train, y_train
    Predictions = knn.predict(x_test)
    True_Labels = y_test

    # Collect classification report between predicted labels and real labels from testing data
    Report= classification_report(True_Labels, Predictions, output_dict=True, zero_division=0)
    if output:
        print(classification_report(True_Labels, Predictions))
    return Report

