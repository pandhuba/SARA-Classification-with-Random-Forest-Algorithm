import numpy as np
import pandas as pd
import mysql.connector as db
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

database = db.connect(host="localhost", user="root", password="", database="sara")
data_train = pd.read_sql('SELECT * FROM data_latih', con=database)

tf = CountVectorizer()
vec = TfidfVectorizer(smooth_idf=False,norm=None)
train_matrix = vec.fit_transform(data_train['comment'].values.astype('U')).toarray()

X = train_matrix
y = data_train['label_comment']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

rf = RandomForestClassifier( n_estimators= 10000,  criterion = 'gini',max_features='sqrt',max_depth=None)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)

from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)
tp = cnf_matrix[0,0]
fp = cnf_matrix[1,0]
fn = cnf_matrix[0,1]
tn = cnf_matrix[1,1]
from sklearn.metrics import classification_report
        
hasil=classification_report(y_test, y_pred)
          
akurasi = (tp+tn)/(tp+fp+fn+tn)
presisi = tp/(tp+fp)
recall = tp/(tp+fn)
print(akurasi)
print(presisi)
print(recall)
