# Required Python Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import json
import os

with open("config.json") as json_file:
	parsed_json = json.load(json_file)

OUTPUT_PATH=''	
for files in parsed_json:
	if(files['python_file']==os.path.realpath(__file__).split("\\")[-1]):
		OUTPUT_PATH = files['xlsxfile']
        
#5 7 3 4 5 1 2 1
HEADERS = ["field_1st_author","field_2nd_author","author_fname",    "author_midname", "auth_suffix", "author_lname_IDF",
           "affl_email","affl_jaccard", "affl_tfidf",  "affl_softtfidf", "affl_dept_jaccard", "affl_org_jaccard","affl_location_jaccard",
           "coauth_lname_shared",    "coauth_lname_idf",    "coauth_jaccard",
           "mesh_shared", "mesh_shared_idf",    "mesh_tree_shared", "mesh_tree_shared_idf",
           "journal_shared_idf", "journal_year", "journal_year_diff",
           "abstract_jaccard",
           "title_jaccard","title_bigram_jaccard", "title_embedding_cosine", "abstract_embedding_cosine", "target"]
target_index = (len(HEADERS))-1

def split_dataset(dataset, train_percentage, feature_headers, target_header):
    """
    Split the dataset with train_percentage
    :param dataset:
    :param train_percentage:
    :param feature_headers:
    :param target_header:
    :return: train_x, test_x, train_y, test_y
    """
 
    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y
 

def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier(n_estimators= 500, min_samples_split= 2, min_samples_leaf= 1, max_features= 'sqrt', max_depth= 50, bootstrap= False)
    #clf = RandomForestClassifier(n_estimators= 1000, min_samples_split= 2, min_samples_leaf= 1, max_features= 'sqrt', max_depth= 100, bootstrap= False)
    clf.fit(features, target)
    return clf

def gradiant_boasted_classifier(features, target):
    """
    To train the gradiant boosted classifier with features and target data
    :param features:
    :param target:
    :return: trained gradiant boosted classifier
    """
    clf = GradientBoostingClassifier(learning_rate = 1,    max_features = None, min_samples_leaf = 0.1,
                                      min_samples_split = 0.1, n_estimators = 200, max_depth = 11)
    clf.fit(features, target)
    return clf

def getrfgbtarrayresult(dataset, modelcode):
    
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[2:target_index], HEADERS[target_index])

    print ("Train_x Shape :: ", train_x.shape)
    print ("Train_y Shape :: ", train_y.shape)
    print ("Test_x Shape :: ", test_x.shape)
    print ("Test_y Shape :: ", test_y.shape)
    
    if(modelcode==1):
        trained_model = random_forest_classifier(train_x, train_y)
    elif(modelcode==2):
        trained_model = gradiant_boasted_classifier(train_x, train_y)
    print ("Trained model :: ", trained_model)
    
    #threshold prediction
    threshold_pred_values = trained_model.predict_proba(test_x)
    #print("Pairwise_result\n",threshold_pred_values)
    predicted = []
    for i in range(0,len(threshold_pred_values)):
        if(threshold_pred_values[i][0]>=0.7):
            predicted.append(0)
        else:
            predicted.append(1)
    
    print("0.7 threshold accuracy ", accuracy_score(test_y, predicted))
    
    predicted = []
    for i in range(0,len(threshold_pred_values)):
        if(threshold_pred_values[i][0]>=0.8):
            predicted.append(0)
        else:
            predicted.append(1)
    
    print("0.8 threshold accuracy ", accuracy_score(test_y, predicted))
    
    predictions = trained_model.predict(test_x)

    #ploting features
    importances = trained_model.feature_importances_
    indices = np.argsort(importances)
    features = HEADERS[2:target_index]
    
    indices_value = []
    for i in range(0,len(indices)):
        indices_value.append(features[indices[i]])
    
    # extracting top 10 features
    impwidth = widthind =  []
    for i in range(16,26):
        impwidth.append(importances[indices[i]])
        
    widthind = indices_value[16:26] 
    
    plt.figure(1)
    plt.barh(range(10), impwidth, color='#5485C0', align='center')
    plt.yticks(range(0,len(widthind)), widthind)
    plt.xlabel('Relative Importance')
    plt.show()
    
    cm = confusion_matrix(test_y, predictions)
    # means all predicted and target value matched then Confusion Matrix size will be 1 X 1 
    if(cm.shape[0]==1):
        TP = len(test_y)
        FN = FP = TN = 0
    else:
        #   Some of them not matched, hence Confusion Matrix Size will be 2 X 2
        result = np.array(cm)
        TP = result[0][0]
        FN = result[0][1]
        FP = result[1][0]
        TN = result[1][1]
    
    S = TP + FN + FP + TN
    
    accuracy = (TP +TN)/S
    precision = TP/(TP+FP)    
    recall = TP/(TP+FN)   
    f1score = 2/(1/precision + 1/recall)

    print('Score ',trained_model.score(test_x,test_y))
    
    print('TP ', TP,' FN', FN,' FP', FP,' TN', TN )
    print ("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
    print ("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    print("Accuracy :: ", accuracy)
    print('Precision ', precision)
    print('Recall ', recall)
    print('F1 - score ', f1score )
    print ("Confusion matrix ", cm)
    
    # Precision Recall Graph
    precisionarray = []
    recallarray = []
    accuracyarray = []
    for l in range(0,11,1):
        predicted = []
        for i in range(0,len(threshold_pred_values)):
            if(threshold_pred_values[i][0]>=l/10):
                predicted.append(0)
            else:
                predicted.append(1)
        cm = confusion_matrix(test_y, predicted)
        #print(cm, len(precisionarray))
        if(cm.shape[0]==1):
            TP = len(test_y)
            FN = FP = TN = 0
        else:
            #   Some of them not matched, hence Confusion Matrix Size will be 2 X 2
            result = np.array(cm)
            TP = result[0][0]
            FN = result[0][1]
            FP = result[1][0]
            TN = result[1][1]
        
        S = TP + FN + FP + TN
        
        accuracy = (TP +TN)/S
        precision = TP/(TP+FP)    
        recall = TP/(TP+FN)   
        f1score = 2/(1/precision + 1/recall)
        accuracyarray.append(accuracy)
        precisionarray.append(precision)
        recallarray.append(recall)
        
    return precisionarray,recallarray
      
def main():

    dataset = pd.read_csv(OUTPUT_PATH, encoding = "ISO-8859-1", error_bad_lines=False)
    
    thresholdarray = []
    for i in range(0,11, 1):
        thresholdarray.append(i/10)
    
    print("\nFOR RANDOM FOREST")
    rfprecisionarray, rfrecallarray = getrfgbtarrayresult(dataset,1)
    print("\nFOR GRADIANT BOOSTED TREE")
    gbtprecisionarray, gbtrecallarray = getrfgbtarrayresult(dataset,2)
    
    plt.plot(thresholdarray, rfprecisionarray, color='b', marker=".") 
    plt.plot(thresholdarray, rfrecallarray, color='g', marker=".")
    plt.plot(thresholdarray, gbtprecisionarray, color='r', marker=".")
    plt.plot(thresholdarray, gbtrecallarray, color='y', marker=".")
    leg = plt.legend(('RFprecision', 'RFrecall','GBTprecision', 'GBTrecall'), frameon=True) 
    leg.get_frame().set_edgecolor('k') 
    plt.xlabel('Threshold') 
    plt.ylabel('Performance')
    
if __name__ == "__main__":
    main()    
