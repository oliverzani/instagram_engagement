import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

#Might need to run these 3 imports/downloads

# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

'''
This script will send each instagram image through google vision's API and return the output in an excel file

Inputs:
    
    1. Replace the path of 'score_df' to the output file of collect_from_insta.py
        
    2. Replace the path of 'google_df' to the output file of google_vision.py
                        
    3. 'like_weight' represents the percentage of the engagement score corresponding to the number of likes
    
    4. 'comment_weight' represents the percentage of the engagement score corresponding to the number of comments
    
    5. 'cap' is a boolean value to determine if you want to include captions in your classification predictors
    
    6. 'goog' is a boolean value to determine if you want to include google vision's output text in your classification predictors
    
Outputs:

    1. The output file with each images Google Vision output and corresponding URL
'''

score_df = pd.read_excel('C:\\Users\\olive\\Documents\\Class\\Medium\\coca_cola.xlsx')
google_df = pd.read_excel('C:\\Users\\olive\\Documents\\Class\\Medium\\google_cola.xlsx')
like_weight = 0.5
comment_weight = 0.5
cap = False
goog = True

'''
other_df = score_df
other_df['Date_ord'] = other_df['Date'].map(dt.datetime.toordinal)
other_df.corr()
'''

#Creating a Binary Version of Engagement
large_df = score_df.merge(google_df, on="URL")
large_df['Engagement'] = like_weight*large_df['Likes_Normalized'] + comment_weight*large_df['Comments_Normalized']
lower_limit = large_df['Engagement'].quantile(q=.33)
upper_limit = large_df['Engagement'].quantile(q=.66)

def categorize(x):
    if x > upper_limit:
        return 2
    elif x < lower_limit:
        return 0
    return 1

large_df['Engagement_Categorical'] = large_df['Engagement'].apply(categorize)


stemmer = SnowballStemmer("english")
def remove_punctuation(s):
    no_punct = ""
    for letter in s:
        if letter not in string_punctuation:
            no_punct += letter
    return no_punct

def words_to_analyze(input_df, caption=False, google=True):
    if caption == True:
        if google == True:
            input_df['text'] = input_df['Caption'] + input_df['Labels']
        else:
            input_df['text'] = input_df['Caption']
    else:
        if google == True:
            input_df['text'] = input_df['Labels']
    return input_df

large_df = words_to_analyze(large_df, caption=cap, google=goog)     
large_df = large_df.dropna(axis = 0)

#Read the text column
string_punctuation = '''()-[]{};:'",<>./?@#$%^&*_~1234567890'''
stop = stopwords.words('english')
large_df.iloc[ :, -1] = large_df.iloc[ :, -1].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

i=0
for row in large_df['text']:
    large_df.iloc[ i, -1] = remove_punctuation(row)
    i=i+1
    
large_df['text'] = large_df['text'].str.replace("!"," !")
large_df['text'] = large_df['text'].apply(word_tokenize)
large_df['text'] = large_df['text'].apply(lambda x: [stemmer.stem(y) for y in x])
large_df['text'] = large_df['text'].apply(lambda x : " ".join(x))
Text_Column = large_df.iloc[ :, -1:]

#Get TFIDF Scores
sklearn_tfidf = TfidfVectorizer(min_df=.01, max_df =.95, stop_words="english",use_idf=True, smooth_idf=False, sublinear_tf=True)
sklearn_representation = sklearn_tfidf.fit_transform(Text_Column.iloc[:, 0].tolist())
Tfidf_Output = pd.DataFrame(sklearn_representation.toarray(), columns=sklearn_tfidf.get_feature_names())

#Append the column to the final dataset
Input = pd.concat([large_df, Tfidf_Output], axis=1)
Input = Input.drop('text', 1)
Input =  pd.concat([Input.Date, Input.iloc[:,9:]], axis=1)


#Removing 
Input = Input.dropna(axis = 0)


X = Input.loc[:, Input.columns != 'Engagement_Categorical']
Y = Input['Engagement_Categorical']

mark = run_classifications(X, Y)
mark_googleonly = run_classifications(X, Y)


import datetime as dt

#Function to run the Classification Models
def run_classifications(X_input, y_input):
    
    X_input['Date'] = pd.to_datetime(X_input['Date'])
    X_input['Date_ord'] = X_input['Date'].map(dt.datetime.toordinal)
    X_input = X_input.drop('Date', 1)
   
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_input)
    X_train, X_test, y_train, y_test = train_test_split(X_std, y_input, test_size = .3, random_state = 20)
    
    result_df = pd.DataFrame(columns=['Precision', 'Recall', 'Accuracy'])
    # Logistic Regression
    logit = LogisticRegression()
    model_lr = logit.fit(X_train, y_train)
    y_test_pred_lr = model_lr.predict(X_test)
    lr_conf_matrix = confusion_matrix(y_test, y_test_pred_lr)
    result_df = result_df.append(pd.DataFrame(data=[[lr_conf_matrix[1,1] / (lr_conf_matrix[1,1] + lr_conf_matrix[0,1]),
                                                     lr_conf_matrix[1,1] / (lr_conf_matrix[1,1] + lr_conf_matrix[1,0]),
                                              accuracy_score(y_test, y_test_pred_lr)]],
                                              index = ["Logistic Regression"], 
                                              columns=['Precision', 'Recall', 'Accuracy']))
    
    # KNN
    for n in range(1,4):
        knn = KNeighborsClassifier(n_neighbors=n)
        model_knn = knn.fit(X_train, y_train)
        y_test_pred_knn = model_knn.predict(X_test)
        knn_conf_matrix = confusion_matrix(y_test, y_test_pred_knn)
        result_df = result_df.append(pd.DataFrame(data=[[knn_conf_matrix[1,1] / (knn_conf_matrix[1,1] + knn_conf_matrix[0,1]),
                                                         knn_conf_matrix[1,1] / (knn_conf_matrix[1,1] + knn_conf_matrix[1,0]),
                                                         accuracy_score(y_test, y_test_pred_knn)]],
                                                  index = ["K Nearest Neighbors n=" + str(n)], 
                                                  columns=['Precision', 'Recall', 'Accuracy']))
    
    # SVM
    svc = SVC()
    model_svc = svc.fit(X_train, y_train)
    y_test_pred_svc = model_svc.predict(X_test)
    svc_conf_matrix = confusion_matrix(y_test, y_test_pred_svc)
    result_df = result_df.append(pd.DataFrame(data=[[svc_conf_matrix[1,1] / (svc_conf_matrix[1,1] + svc_conf_matrix[0,1]),
                                                     svc_conf_matrix[1,1] / (svc_conf_matrix[1,1] + svc_conf_matrix[1,0]),
                                                     accuracy_score(y_test, y_test_pred_svc)]],
                                              index = ["Support Vector Machine"], 
                                              columns=['Precision', 'Recall', 'Accuracy']))
    
    # Random Forests
    randomforest = RandomForestClassifier(random_state=5)
    model_rf = randomforest.fit(X_train, y_train)
    y_test_pred_rf = model_rf.predict(X_test)
    rf_conf_matrix = confusion_matrix(y_test, y_test_pred_rf)
    result_df = result_df.append(pd.DataFrame(data=[[rf_conf_matrix[1,1] / (rf_conf_matrix[1,1] + rf_conf_matrix[0,1]),
                                                     rf_conf_matrix[1,1] / (rf_conf_matrix[1,1] + rf_conf_matrix[1,0]),
                                                     accuracy_score(y_test, y_test_pred_rf)]],
                                              index = ["Random Forest"], 
                                              columns=['Precision', 'Recall', 'Accuracy']))
    
    randomforest_balanced = RandomForestClassifier(random_state=5, class_weight="balanced")
    model_rfb = randomforest_balanced.fit(X_train, y_train)
    y_test_pred_rfb = model_rfb.predict(X_test)
    rfb_conf_matrix = confusion_matrix(y_test, y_test_pred_rfb)
    result_df = result_df.append(pd.DataFrame(data=[[rfb_conf_matrix[1,1] / (rfb_conf_matrix[1,1] + rfb_conf_matrix[0,1]),
                                                     rfb_conf_matrix[1,1] / (rfb_conf_matrix[1,1] + rfb_conf_matrix[1,0]),
                                                     accuracy_score(y_test, y_test_pred_rfb)]],
                                              index = ["Random Forest Balanced"], 
                                              columns=['Precision', 'Recall', 'Accuracy']))
    
    # ANN
    for iterr in range(200,1000,100):
        ann = MLPClassifier(max_iter=iterr)
        model_ann = ann.fit(X_train, y_train)
        y_test_pred_ann = model_ann.predict(X_test)
        ann_conf_matrix = confusion_matrix(y_test, y_test_pred_ann)
        result_df = result_df.append(pd.DataFrame(data=[[ann_conf_matrix[1,1] / (ann_conf_matrix[1,1] + ann_conf_matrix[0,1]),
                                                         ann_conf_matrix[1,1] / (ann_conf_matrix[1,1] + ann_conf_matrix[1,0]),
                                                         accuracy_score(y_test, y_test_pred_ann)]],
                                                  index = ["Artificial Neural Networks iter=" + str(iterr)], 
                                                  columns=['Precision', 'Recall', 'Accuracy']))    
    return result_df

