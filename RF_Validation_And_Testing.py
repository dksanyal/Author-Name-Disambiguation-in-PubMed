# Required Python Packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

#6 7 3 4 5 1 2 1
HEADERS = ["field_1st_author","field_2nd_author","author_fname",    "author_midname", "auth_suffix", "author_lname_IDF",
           "affl_email","affl_jaccard", "affl_tfidf",  "affl_softtfidf", "affl_dept_jaccard", "affl_org_jaccard","affl_location_jaccard",
           "coauth_lname_shared",    "coauth_lname_idf",    "coauth_jaccard", "coauth_lname_finitial_jaccard",
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
    clf = RandomForestClassifier(n_estimators= 500, min_samples_split= 2, min_samples_leaf= 1, max_features= 'sqrt', max_depth= 100, bootstrap= False)
    clf.fit(features, target)
    return clf

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
      
def main():

    dataset = pd.read_csv(OUTPUT_PATH, encoding = "ISO-8859-1", error_bad_lines=False)
    #dataset_statistics(dataset)
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[2:target_index], HEADERS[target_index])

    training_x, validation_x, training_y, validation_y = train_test_split(train_x, train_y,train_size=0.7)
    print ("Train_x Shape :: ", train_x.shape)
    print ("Train_y Shape :: ", train_y.shape)
    print ("Validation_x Shape :: ", validation_x.shape)
    print ("Validation_y Shape :: ", validation_y.shape)
    print ("Test_x Shape :: ", test_x.shape)
    print ("Test_y Shape :: ", test_y.shape)
    
    #for validation
    trained_model = random_forest_classifier(training_x, training_y)
    print ("Trained model :: ", trained_model)
    
    #threshold prediction
    threshold_pred_values = trained_model.predict_proba(validation_x)
    predicted = []
    for i in range(0,len(threshold_pred_values)):
        if(threshold_pred_values[i][0]>=0.7):
            predicted.append(0)
        else:
            predicted.append(1)
    
    print("0.7 threshold validation accuracy ", accuracy_score(validation_y, predicted))
    
    predicted = []
    for i in range(0,len(threshold_pred_values)):
        if(threshold_pred_values[i][0]>=0.8):
            predicted.append(0)
        else:
            predicted.append(1)
    
    print("0.8 threshold validation accuracy ", accuracy_score(validation_y, predicted))
    
    predictions = trained_model.predict(validation_x)

    
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
    
    cm = confusion_matrix(validation_y, predictions)
    # means all predicted and target value matched then Confusion Matrix size will be 1 X 1 
    if(cm.shape[0]==1):
        TP = len(validation_y)
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

    print('Score ',trained_model.score(validation_x,validation_y))
    
    print('TP ', TP,' FN', FN,' FP', FP,' TN', TN )
    print ("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
    print ("Validation Accuracy  :: ", accuracy_score(validation_y, predictions))
    print("Validation Accuracy Score :: ", accuracy)
    print('Validation Precision ', precision)
    print('Validation Recall ', recall)
    print('Validation F1 - score ', f1score )
    print ("Validation Confusion matrix ", cm)
    
    #for testing
    print("TESTING RESULTS")
    threshold_pred_values = trained_model.predict_proba(test_x)
    predicted = []
    for i in range(0,len(threshold_pred_values)):
        if(threshold_pred_values[i][0]>=0.7):
            predicted.append(0)
        else:
            predicted.append(1)
    
    print("0.7 threshold test accuracy ", accuracy_score(test_y, predicted))
    
    predicted = []
    for i in range(0,len(threshold_pred_values)):
        if(threshold_pred_values[i][0]>=0.8):
            predicted.append(0)
        else:
            predicted.append(1)
    
    print("0.8 threshold test accuracy ", accuracy_score(test_y, predicted))
    
    predictions = trained_model.predict(test_x)

    
    
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
    print ("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    print("Accuracy Test Score:: ", accuracy)
    print('Test Precision ', precision)
    print('Test Recall ', recall)
    print('Test F1 - score ', f1score )
    print ("Test Confusion matrix ", cm)
    
    
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
    thresholdarray = []
    for i in range(0,11, 1):
        thresholdarray.append(i/10)
    
    plt.plot(thresholdarray, precisionarray, color='b', marker=".") 
    plt.plot(thresholdarray, recallarray, color='g', marker=".") 
    plt.plot(thresholdarray, accuracyarray, color='r', marker=".") 
    leg = plt.legend(('Precision', 'Recall','Accuracy'), frameon=True) 
    leg.get_frame().set_edgecolor('k') 
    plt.xlabel('Threshold') 
    plt.ylabel('Performance')
    
    
if __name__ == "__main__":
    main()    
