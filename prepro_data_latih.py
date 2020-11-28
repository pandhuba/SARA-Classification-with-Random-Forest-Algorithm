from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize 
import mysql.connector as db
import pandas as pan
import sqlalchemy
import re

database = db.connect(host="localhost", user="root", password="", database="sara")
data = pan.read_sql('SELECT * FROM dataset', con=database)

data['comment'] = data['comment'].apply(lambda x: " ".join(x.lower() for x in x.split())) 
data['comment'] = data['comment'].str.replace("[^a-zA-Z]", " ")

#tokenisasi
tokenisasi=[]
for row in data['comment']: 
    token = word_tokenize(row)
    tokenisasi.append(token)
    
table={} 
with open('spelling_word.txt','r') as syn:
    for row in syn:
        match=re.match(r'(\w+)\s+=\s+(.+)',row)
        if match:
            primary,synonyms=match.groups() 
            synonyms=[synonym.lower() for synonym in synonyms.split()] 
            for synonym in synonyms:
                table[synonym]=primary.lower()
                
spelling=[]
for idx,value in enumerate (tokenisasi):
    temp=[]
    for idy,value1 in enumerate (value):
        temp.append(''.join(table.get(word.lower(),word) for word in re.findall(r'(\W+|\w+)',value1)))
    spelling.append(temp)
    
#stopword
stop_factory = StopWordRemoverFactory()
data_stopword = stop_factory.get_stop_words()
stopword = stop_factory.create_stop_word_remover()
stopword_removal=[]
for idx,value in enumerate (spelling):
    temp=[]
    for idy,value1 in enumerate (value):
        temp.append(stopword.remove(value1))
    stopword_removal.append(temp)

#stemming:
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stemming=[]
for idx,value in enumerate (stopword_removal):
    temp=[]
    for idy,value1 in enumerate (value):
        temp.append(stemmer.stem(value1))
    stemming.append(temp)  
    
hasil_prepro=[]
for idx,value in enumerate (stemming):
    punctuations = ''' '''
    no_punct = ""
    for idy,value1 in enumerate (value):
        if value1 not in punctuations:
            no_punct = no_punct + value1 + ' '
        k = no_punct
    hasil_prepro.append(k)

data['comment'] = pan.Series(hasil_prepro)
data['label_comment']= data['label'].factorize()[0]
label_comment=data[['label','label_comment']].drop_duplicates().sort_values('label_comment')
label1 = dict(label_comment.values)
label2 = dict(label_comment[['label_comment', 'label']].values)

db_username = 'root'
db_password = ''
db_ip       = 'localhost'
db_name     = 'sara'
db_connection = sqlalchemy.create_engine('mysql+mysqlconnector://{0}:{1}@{2}/{3}'.format(db_username, db_password, db_ip, db_name))

data.to_sql(con=db_connection, name='data_latih', if_exists='replace',index=False)
 