from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import pandas as pd
import numpy as np
from skfeature.function.similarity_based import fisher_score #import fischer score features selected algorithm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif #import info gain features selected algorithm
from sklearn.feature_selection import SelectKBest
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV #grid class for tuning each algorithm
import timeit


main = tkinter.Tk()
main.title("Darknet Classification") #designing main screen
main.geometry("1300x1200")


global pad_data,os_data,ab_cls
global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca,training_time, testing_time,selected_features
global accuracy, precision, recall, fscore, values,algorithm, predict,no_def_X,no_def_Y,pad_X,pad_Y,info_gain
accuracy = []
precision = []
recall = []
fscore = []
train_time = []
test_time = []

def uploadDataset():
    global filename, dataset, labels, values,no_def_X,no_def_Y,pad_X,pad_Y,info_gain,selected_features
    global dataset, X, Y,no_def_X,no_def_Y,pad_X,pad_Y,info_gain,selected_features
    global X_train, X_test, y_train, y_test, pca, scaler,labels,training_time, testing_time
    filename1 = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,'Dataset loaded\n\n')
    with open("Dataset/X_train_NoDef.pkl", 'rb') as handle:
        no_def_X = np.array(pickle.load(handle, encoding='latin1'))
    handle.close()
    no_def_X = no_def_X[:,0:300]
    no_def_X = pd.DataFrame(no_def_X)
    text.insert(END,"No Defence Dataset"+"\n")
    text.insert(END,no_def_X)

    filename2 = filedialog.askopenfilename(initialdir = "Dataset")
    with open("Dataset/y_train_NoDef.pkl", 'rb') as handle:
        no_def_Y = np.array(pickle.load(handle, encoding='latin1'))
    handle.close()
    text.insert(END,"\n Tor Traffic Class Labels"+"\n")
    text.insert(END,no_def_Y)

    filename3 = filedialog.askopenfilename(initialdir = "Dataset")
    with open("Dataset/WTFPAD.pkl", 'rb') as handle:
        pad_X = np.array(pickle.load(handle, encoding='latin1'))
    handle.close()
    pad_X = pad_X[:,0:300]
    pad_X = pd.DataFrame(pad_X)
    text.insert(END,"\n WTFPAD Dataset"+"\n")
    text.insert(END,pad_X)

    
    filename4 = filedialog.askopenfilename(initialdir = "Dataset")
    with open("Dataset/label_WTFPAD.pkl", 'rb') as handle:
        pad_Y = np.array(pickle.load(handle, encoding='latin1'))
    handle.close()
    text.insert(END,"WTFPAD Traffic Class Labels"+"\n")
    text.insert(END,pad_Y)

    selected_features = SelectKBest(mutual_info_classif, k=50)#using info gain select 50 features
    no_def_X = selected_features.fit_transform(no_def_X, no_def_Y)
    info_gain = mutual_info_classif(no_def_X, no_def_Y)
    info_gain = pd.Series(info_gain)
    info_gain.plot(kind='bar', color='teal', figsize=(12,4))
    plt.xlabel("Feature Name")
    plt.ylabel("Importance")
    plt.title("Information Gain Features Selected")
    plt.xticks(rotation=90)
    plt.show()

    corr_matrix = pd.DataFrame(no_def_X).corr()
    corr_matrix.plot(kind='bar', figsize=(12,4))
    plt.show()

def graph():
    global dataset, X, Y,no_def_X,no_def_Y,pad_X,pad_Y,info_gain,selected_features,pad_X_train, pad_X_test, pad_y_train, pad_y_test
    global X_train, X_test, y_train, y_test, pca, scaler,labels,training_time, testing_time,def_X_train, def_X_test, def_y_train, def_y_test

    text.delete('1.0', END)

    #get correlation features selection
    corr_matrix = pd.DataFrame(no_def_X).corr()
    corr_matrix.plot(kind='bar', figsize=(12,4))
    plt.show()
            
    ranks = fisher_score.fisher_score(no_def_X, no_def_Y)
    ranks = pd.Series(ranks)
    ranks.plot(kind='bar', color='teal', figsize=(12,4))
    plt.xlabel("Feature Name")
    plt.ylabel("Importance")
    plt.title("Fisher Score Features Selected")
    plt.xticks(rotation=90)
    plt.show()
        
def processDataset():
    global dataset, X, Y,no_def_X,no_def_Y,pad_X,pad_Y,info_gain,selected_features,pad_X_train, pad_X_test, pad_y_train, pad_y_test
    global X_train, X_test, y_train, y_test, pca, scaler,labels,training_time, testing_time,def_X_train, def_X_test, def_y_train, def_y_test

    text.delete('1.0', END)

    pad_X = selected_features.transform(pad_X)
    indices = np.arange(no_def_X.shape[0])
    np.random.shuffle(indices)
    no_def_X = no_def_X[indices]
    no_def_Y = no_def_Y[indices]

    indices = np.arange(pad_X.shape[0])
    np.random.shuffle(indices)
    pad_X = pad_X[indices]
    pad_Y = pad_Y[indices]
    text.insert(END,"Dataset Preprocessing Completed"+"\n")

    #split both No-Defence and PAD dataset into train and test
    def_X_train, def_X_test, def_y_train, def_y_test = train_test_split(no_def_X, no_def_Y, test_size=0.2)
    pad_X_train, pad_X_test, pad_y_train, pad_y_test = train_test_split(pad_X, pad_Y, test_size=0.2)
    text.insert(END,"Train & Test Dataset Split"+"\n")
    text.insert(END,"80% audio features used to train algorithms : "+str(def_X_train.shape[0])+"\n")
    text.insert(END,"20% audio features used to rest algorithms : "+str(def_X_test.shape[0])+"\n")

def calculateMetrics(algorithm, predict, y_test, training_time, testing_time):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    train_time.append(training_time)
    test_time.append(testing_time)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")    
    

def calculateMetric(algorithm, predict, y_test, training_time, testing_time):
    labels = ['WPPAD', 'Original']
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    train_time.append(training_time)
    test_time.append(testing_time)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")    
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 3)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.xticks(rotation=90)
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    
       

def trainKNN():
    global dataset, X, Y,no_def_X,no_def_Y,pad_X,pad_Y,info_gain,selected_features
    global X_train, X_test, y_train, y_test, pca, scaler,labels,training_time, testing_time
    text.delete('1.0', END)

    tuning_param = {'n_neighbors' : [1], 'p' : [1]}
    knn_no_defence = GridSearchCV(KNeighborsClassifier(), tuning_param, cv=5)#defining svm with tuned parameters
    start = timeit.default_timer()
    knn_no_defence.fit(no_def_X, no_def_Y)#now train KNN
    end = timeit.default_timer()
    predict = knn_no_defence.predict(def_X_test) #perfrom prediction on test data
    end1 = timeit.default_timer()
    calculateMetrics("Original (No Defence) KNN", predict, def_y_test, (end - start), (end1 - end))


def trainRF():
    global dataset, X, Y,no_def_X,no_def_Y,pad_X,pad_Y,info_gain,selected_features
    global X_train, X_test, y_train, y_test, pca, scaler,labels,training_time, testing_time
    text.delete('1.0', END)

    #train Random Forest algorithm on No-Defence Dataset
    #defining Random Forest tuning parameters
    tuning_param = {'n_estimators' : [90]}
    rf_no_defence = GridSearchCV(RandomForestClassifier(), tuning_param, cv=5)#defining svm with tuned parameters
    start = timeit.default_timer()
    rf_no_defence.fit(no_def_X, no_def_Y)#now train KNN
    end = timeit.default_timer()
    predict = rf_no_defence.predict(def_X_test) #perfrom prediction on test data
    end1 = timeit.default_timer()
    calculateMetrics("Original (No Defence) Random Forest", predict, def_y_test, (end - start), (end1 - end))


def trainSVM():
    global dataset, X, Y,no_def_X,no_def_Y,pad_X,pad_Y,info_gain,selected_features,def_X_test
    global X_train, X_test, y_train, y_test, pca, scaler,labels,training_time, testing_time
    text.delete('1.0', END)

    #train SVM algorithm on No-Defence Dataset
    #defining SVM tuning parameters
    tuning_param = {'C' : [100], 'kernel': ['rbf']}
    svm_no_defence = GridSearchCV(svm.SVC(), tuning_param, cv=5)#defining svm with tuned parameters
    start = timeit.default_timer()
    svm_no_defence.fit(no_def_X, no_def_Y)#now train KNN
    end = timeit.default_timer()
    predict = svm_no_defence.predict(def_X_test) #perfrom prediction on test data
    end1 = timeit.default_timer()
    calculateMetrics("Original (No Defence) SVM", predict, def_y_test, (end - start), (end1 - end))

def trainRKNN():
    global dataset, X, Y,no_def_X,no_def_Y,pad_X,pad_Y,info_gain,selected_features
    global X_train, X_test, y_train, y_test, pca, scaler,labels,training_time, testing_time
    text.delete('1.0', END)

    #train KNN algorithm on WTFPAD Dataset
    #defining KNN tuning parameters
    tuning_param = {'n_neighbors' : [1], 'p' : [1]}
    knn_pad = GridSearchCV(KNeighborsClassifier(), tuning_param, cv=5)#defining svm with tuned parameters
    start = timeit.default_timer()
    knn_pad.fit(pad_X[0:8000], pad_Y[0:8000])#now train KNN
    end = timeit.default_timer()
    predict = knn_pad.predict(pad_X_test) #perfrom prediction on test data
    end1 = timeit.default_timer()
    calculateMetrics("WTFPAD Defence KNN", predict, pad_y_test, (end - start), (end1 - end))


def trainRRF():
    global dataset, X, Y,no_def_X,no_def_Y,pad_X,pad_Y,info_gain,selected_features
    global X_train, X_test, y_train, y_test, pca, scaler,labels,training_time, testing_time
    text.delete('1.0', END)

    #train Random Forest algorithm on WTFPAD Dataset
    #defining Random Forest tuning parameters
    tuning_param = {'n_estimators' : [100]}
    rf_pad = GridSearchCV(RandomForestClassifier(), tuning_param, cv=5)#defining svm with tuned parameters
    start = timeit.default_timer()
    rf_pad.fit(pad_X[0:8000], pad_Y[0:8000])#now train KNN
    end = timeit.default_timer()
    predict = rf_pad.predict(pad_X_test) #perfrom prediction on test data
    end1 = timeit.default_timer()
    calculateMetrics("WTFPAD Defence Random Forest", predict, pad_y_test, (end - start), (end1 - end))



def trainRSVM():
    global dataset, X, Y,no_def_X,no_def_Y,pad_X,pad_Y,info_gain,selected_features,def_X_test
    global X_train, X_test, y_train, y_test, pca, scaler,labels,training_time, testing_time
    text.delete('1.0', END)

    #train SVM algorithm on WTFPAD Dataset
    #defining SVM tuning parameters
    tuning_param = {'C' : [100], 'kernel': ['rbf']}
    svm_pad = GridSearchCV(svm.SVC(), tuning_param, cv=5)#defining svm with tuned parameters
    start = timeit.default_timer()
    svm_pad.fit(pad_X, pad_Y)#now train KNN
    end = timeit.default_timer()
    predict = svm_pad.predict(pad_X_test) #perfrom prediction on test data
    end1 = timeit.default_timer()
    calculateMetrics("WTFPAD Defence SVM", predict, pad_y_test, (end - start), (end1 - end))

def MergeData():
    global pad_data,os_data
    global dataset, X, Y,no_def_X,no_def_Y,pad_X,pad_Y,info_gain,selected_features
    global X_train, X_test, y_train, y_test, pca, scaler,labels,training_time, testing_time
    text.delete('1.0', END)
    
    pad_data = pad_X[0:482]
    os_data = pd.read_csv("Dataset/OS.csv")#read OS onion service dataset
    os_data.fillna(0, inplace = True)
    os_data = os_data.values
    pad_data = pad_data[:,0:6]
    os_data = os_data[:,1:7]
    X = []
    Y = []
    for i in range(len(pad_data)):#merge pad and OS data
        X.append(pad_data[i])
        Y.append(0)
    for i in range(len(os_data)):
        X.append(os_data[i])
        Y.append(1)
    X = np.asarray(X)
    Y = np.asarray(Y)
    text.insert(END,"OS & WTFPAD Data merging completed"+"\n")

def trainWRKNN():
    global pad_data,os_data
    global dataset, X, Y,no_def_X,no_def_Y,pad_X,pad_Y,info_gain,selected_features
    global X_train, X_test, y_train, y_test, pca, scaler,labels,training_time, testing_time
    text.delete('1.0', END)
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    #train KNN algorithm on WTFPAD Dataset
    #defining KNN tuning parameters
    tuning_param = {'n_neighbors' : [4], 'p' : [1]}
    knn = GridSearchCV(KNeighborsClassifier(), tuning_param, cv=5)#defining svm with tuned parameters
    start = timeit.default_timer()
    knn.fit(X_train, y_train)#now train KNN
    end = timeit.default_timer()
    predict = knn.predict(X_test) #perfrom prediction on test data
    end1 = timeit.default_timer()
    calculateMetric("WTFPAD KNN Top 6 Features", predict, y_test, (end - start), (end1 - end))

def trainWRRF():
    global pad_data,os_data
    global dataset, X, Y,no_def_X,no_def_Y,pad_X,pad_Y,info_gain,selected_features
    global X_train, X_test, y_train, y_test, pca, scaler,labels,training_time, testing_time
    text.delete('1.0', END)

    #train Random Forest algorithm on WTFPAD Dataset
    #defining Random Forest tuning parameters
    tuning_param = {'n_estimators' : [5], 'max_depth': [5]}
    rf = GridSearchCV(RandomForestClassifier(), tuning_param, cv=5)#defining svm with tuned parameters
    start = timeit.default_timer()
    rf.fit(X_train, y_train)#now train KNN
    end = timeit.default_timer()
    predict = rf.predict(X_test) #perfrom prediction on test data
    end1 = timeit.default_timer()
    calculateMetric("WTFPAD Random Forest Top 6 Features", predict, y_test, (end - start), (end1 - end))

def trainWRSVM():
    global pad_data,os_data
    global dataset, X, Y,no_def_X,no_def_Y,pad_X,pad_Y,info_gain,selected_features,def_X_test
    global X_train, X_test, y_train, y_test, pca, scaler,labels,training_time, testing_time
    text.delete('1.0', END)

    tuning_param = {'C' : [7], 'kernel': ['rbf']}
    svm_cls = GridSearchCV(svm.SVC(), tuning_param, cv=5)#defining svm with tuned parameters
    start = timeit.default_timer()
    svm_cls.fit(X_train, y_train)#now train KNN
    end = timeit.default_timer()
    predict = svm_cls.predict(X_test) #perfrom prediction on test data
    end1 = timeit.default_timer()
    calculateMetric("WTFPAD SVM Top 6 Features", predict, y_test, (end - start), (end1 - end))

def trainXGB():
    global pad_data,os_data,ab_cls
    global dataset, X, Y,no_def_X,no_def_Y,pad_X,pad_Y,info_gain,selected_features,def_X_test
    global X_train, X_test, y_train, y_test, pca, scaler,labels,training_time, testing_time
    text.delete('1.0', END)

    #extension XGBoost training on merge dataset
    ab_cls = AdaBoostClassifier(n_estimators=100)
    start = timeit.default_timer()
    ab_cls.fit(X_train, y_train)#now train KNN
    end = timeit.default_timer()
    predict = ab_cls.predict(X_test) #perfrom prediction on test data
    end1 = timeit.default_timer()
    calculateMetric("WTFPAD Extension AdaBoost Top 6 Features", predict, y_test, (end - start), (end1 - end))

def predict():
    global pad_data,os_data,ab_cls
    global dataset, X, Y,no_def_X,no_def_Y,pad_X,pad_Y,info_gain,selected_features,def_X_test
    global X_train, X_test, y_train, y_test, pca, scaler,labels,training_time, testing_time
    text.delete('1.0', END)

    filename = filedialog.askopenfilename(initialdir = "Dataset")
    #read test data features
    testData = pd.read_csv(filename)
    testData = testData.values
    #using extension AdaBoost classify network type
    predict = ab_cls.predict(testData)
    labels = ['Tor Service', 'Onion Service']
    #loop and display predicted values
    for i in range(len(testData)):
        text.insert(END,"Test Network Data : "+str(testData[i])+" Predicted As ====> "+labels[predict[i]]+"\n")

font = ('times', 16, 'bold')
title = Label(main, text='Darknet Traffic Analysis: Investigating the Impact of Modified Tor Traffic on Onion Service Traffic Classification')
title.config(bg='VioletRed4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=27,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Correlation Features Selection", command=graph)
processButton.place(x=160,y=100)
processButton.config(font=font1)

processButton = Button(main, text="Preprocess & Split Dataset", command=processDataset)
processButton.place(x=430,y=100)
processButton.config(font=font1)

KnnButton = Button(main, text="Original KNN", command=trainKNN)
KnnButton.place(x=660,y=100)
KnnButton.config(font=font1)

RFButton = Button(main, text="Original Random Forest", command=trainRF)
RFButton.place(x=800,y=100)
RFButton.config(font=font1)

SVMButton = Button(main, text="Original SVM", command=trainSVM)
SVMButton.place(x=1010,y=100)
SVMButton.config(font=font1)

rRFButton = Button(main, text="Defence Random Forest", command=trainRRF)
rRFButton.place(x=1160,y=100)
rRFButton.config(font=font1)

rKnnButton = Button(main, text="Defence KNN", command=trainRKNN)
rKnnButton.place(x=10,y=150)
rKnnButton.config(font=font1)


rSVMButton = Button(main, text="Defence SVM", command=trainRSVM)
rSVMButton.place(x=140,y=150)
rSVMButton.config(font=font1)

MButton = Button(main, text="Merge Dataset", command=MergeData)
MButton.place(x=300,y=150)
MButton.config(font=font1)


WKnnButton = Button(main, text="WTFPAD KNN", command=trainWRKNN)
WKnnButton.place(x=450,y=150)
WKnnButton.config(font=font1)

WRFButton = Button(main, text="WTFPAD Random Forest", command=trainWRRF)
WRFButton.place(x=600,y=150)
WRFButton.config(font=font1)

WSVMButton = Button(main, text="WTFPAD SVM", command=trainWRSVM)
WSVMButton.place(x=830,y=150)
WSVMButton.config(font=font1)

WrRFButton = Button(main, text="WTFPAD XGBOOST", command=trainXGB)
WrRFButton.place(x=980,y=150)
WrRFButton.config(font=font1)

PButton = Button(main, text="Predict", command=predict)
PButton.place(x=1180,y=150)
PButton.config(font=font1)

main.config(bg='VioletRed1')
main.mainloop()



