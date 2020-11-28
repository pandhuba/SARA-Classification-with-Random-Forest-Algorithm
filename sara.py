from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from flask import Flask, render_template, url_for, session, request, redirect, flash
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from werkzeug.utils import secure_filename
from sklearn.tree import export_graphviz
from sklearn.externals import joblib
from nltk.tokenize import word_tokenize
from datetime import datetime, date
from urllib.request import urlopen
from IPython.display import Image
from bs4 import BeautifulSoup
import mysql.connector as db
import pandas as pan
import numpy as np
import sqlalchemy
import bcrypt
import re
import os

app = Flask(__name__)
app.config['SECRET_KEY']='panda'

database = db.connect(host="localhost", user="root", password="", database="sara")

@app.route('/')
@app.route('/index')
def index():
    return render_template('login.html')

@app.route('/login_admin')
def login_admin():
    return render_template('login_admin.html')

@app.route('/proseslogin_admin', methods=['GET', 'POST'])
def proseslogin_admin():
    if request.method == 'POST':
        username = request.form['username']
        
        curl = database.cursor()
        curl.execute("SELECT * FROM admin WHERE username=%s",(username,))
        user = curl.fetchone()
        curl.close()

        if len(user) > 0:
            if request.form['password'] == user[2]:
                session['username'] = user[1]
                return redirect(url_for('index_admin'))
            else:
                return "ERROR USERNAME DAN PASSWORD TIDAK SAMA"
        else:
            return "ERROR PENGGUNA TIDAK DITEMUKAN"
    else:
        return render_template("login_admin.html")

@app.route('/proseslogin_member', methods=['GET', 'POST'])
def proseslogin_member():
	if request.method == 'POST':
		username = request.form['username']

		curl = database.cursor()
		curl.execute("SELECT * FROM member WHERE username=%s OR no_hp=%s",(username,username,))
		user = curl.fetchone()
		curl.close()

		if len(user) > 0:
			if request.form['password'] == user[1]:
				session['username'] = user[0]
				return redirect(url_for('index_member'))
			else:
				return "ERROR USERNAME/NO_HANDPHONE DAN PASSWORD TIDAK SAMA"
		else:
			return "ERROR PENGGUNA TIDAK DITEMUKAN"
	else:
		return render_template("login_member.html")

@app.route('/daftar', methods=['GET','POST'])
def daftar():
	if request.method == 'GET':
		return render_template("daftar.html")
	else:
		nama = request.form['nama_lengkap']
		no_hp = request.form['no_hp']
		username = request.form['username']
		password = request.form['password']

		cur = database.cursor()
		cur.execute("SELECT * FROM member WHERE username=%s OR no_hp=%s",(username,no_hp,))
		user = cur.fetchall()

		if len(user) > 0:
			return "USERNAME / NO HANDPHONE SUDAH DIGUNAKAN"
		else:
			cur.execute("INSERT INTO member (username, password, nama_lengkap, no_hp) VALUES (%s,%s,%s,%s)",(username,password,nama,no_hp,))
			database.commit()
			return redirect(url_for('index'))

#===============================================<><><>=================================================================================

@app.route('/index_admin')
def index_admin():
	if 'username' in session:
		data={}
		user = session['username']
		data['username'] = user

		return render_template('admin/index_admin.html', text=data)
	else:
		return redirect(url_for('login_admin'))

@app.route('/data_admin')
def data_admin():
	if 'username' in session:
		data={}
		user = session['username']
		data['username'] = user

		cur = database.cursor()
		cur.execute("SELECT * FROM admin")
		rv = cur.fetchall()
		cur.close()
		data['select'] = rv
		return render_template('admin/data_admin.html', text=data)
	else:
		return redirect(url_for('login_admin'))

@app.route('/tambah_admin')
def tambah_admin():
	if 'username' in session:
		data={}
		user = session['username']
		data['username'] = user
		return render_template('admin/tambah_admin.html', text=data)
	else :
		return redirect(url_for('login_admin'))

@app.route('/simpan_admin', methods=['POST'])
def simpan_admin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cur = database.cursor()
        cur.execute("SELECT * FROM admin WHERE username=%s",(username,))
        user = cur.fetchall()

        if len(user) == 0:
        	cur.execute("INSERT INTO admin (username, password) VALUES (%s,%s)",(username,password,))
	        database.commit()
	        return redirect(url_for('data_admin'))
        else:
            return "USERNAME SUDAH DIGUNAKAN"
        
@app.route('/edit_admin/<string:id_admin>', methods=['GET'])
def edit_admin(id_admin):
	if 'username' in session:
		data={}
		user = session['username']
		data['username'] = user

		cur = database.cursor()
		cur.execute("SELECT * FROM admin WHERE id_admin=%s", (id_admin,))
		rv = cur.fetchone()
		cur.close()
		data['select'] = rv
		return render_template('admin/edit_admin.html', text=data)
	else :
		return redirect(url_for('login_admin'))

@app.route('/update_admin', methods=['POST'])
def update_admin():
	if request.method == 'POST':
		username = request.form['username']
		password = request.form['password']
		id_admin = request.form['id_admin']

		cur = database.cursor()
		cur.execute("SELECT * FROM admin WHERE username=%s",(username,))
		user = cur.fetchall()

		if len(user) == 0:
			sql = "UPDATE admin SET username=%s, password=%s WHERE id_admin=%s"
			data = (username, password, id_admin,)
			cur.execute(sql, data)
			database.commit()
			return redirect(url_for('data_admin'))
		else:
			return "USERNAME SUDAH DIGUNAKAN"

@app.route('/hapus_admin/<string:id_admin>', methods=["GET"])
def hapus_admin(id_admin):
    cur = database.cursor()
    cur.execute("DELETE FROM admin WHERE id_admin=%s", (id_admin,))
    database.commit()
    return redirect(url_for('data_admin'))

@app.route('/data_member')
def data_member():
	if 'username' in session:
		data = {}
		user = session['username']
		data['username'] = user

		cur = database.cursor()
		cur.execute("SELECT * FROM member")
		rv = cur.fetchall()
		cur.close()
		data['select'] = rv
		return render_template('admin/data_member.html', members=data)
	else:
		return redirect(url_for('login_admin'))

@app.route('/edit_member/<string:username>', methods=['GET'])
def edit_member(username):
	if 'username' in session:
		data={}
		user = session['username']
		data['username'] = user

		cur = database.cursor()
		cur.execute("SELECT * FROM member WHERE username=%s", (username,))
		rv = cur.fetchone()
		cur.close()
		data['select'] = rv
		return render_template('admin/edit_member.html', text=data)
	else :
		return redirect(url_for('login_admin'))

@app.route('/update_member', methods=['POST'])
def update_member():
	if request.method == 'POST':
		nama = request.form['nama']
		no_hp = request.form['no_hp']
		username = request.form['username']
		password = request.form['password']

		cur = database.cursor()
		cur.execute("SELECT * FROM member WHERE username=%s OR no_hp=%s",(username,no_hp,))
		user = cur.fetchall()

		if len(user) == 0:
			sql = "UPDATE member SET password=%s nama_lengkap=%s, no_hp=%s WHERE username=%s"
			data = (password, nama, no_hp, username)
			cur.execute(sql, data)
			database.commit()
			return redirect(url_for('data_member'))
		else:
			return "NO HANDPHONE SUDAH DIGUNAKAN"

@app.route('/hapus_member/<string:username>', methods=["GET"])
def hapus_member(username):
    cur = database.cursor()
    cur.execute("DELETE FROM member WHERE username=%s", (username,))
    database.commit()
    return redirect(url_for('data_member'))

@app.route('/data_dataset_admin')
def data_dataset_admin():
	if 'username' in session:
		data={}
		user = session['username']
		data['username'] = user

		cur = database.cursor()
		cur.execute("SELECT * FROM dataset")
		rv = cur.fetchall()
		cur.close()
		data['select'] = rv
		return render_template('admin/data_dataset_admin.html', datasets=data)
	else :
		return redirect(url_for('login_admin'))

@app.route('/tambah_dataset')
def tambah_dataset():
	if 'username' in session:
		data={}
		user = session['username']
		data['username'] = user
		return render_template('admin/tambah_dataset.html', text=data)
	else :
		return redirect(url_for('login_admin'))

@app.route('/simpan_dataset', methods=['POST'])
def simpan_dataset():
    if request.method == 'POST':
        comment = request.form['comment']
        label = request.form['label']

        cur = database.cursor()
        cur.execute("INSERT INTO dataset (comment, label) VALUES (%s,%s)",(comment,label,))
        database.commit()
        return redirect(url_for('data_dataset_admin'))

@app.route('/edit_dataset/<string:id>', methods=['GET'])
def edit_dataset(id):
	if 'username' in session:
		data={}
		user = session['username']
		data['username'] = user

		cur = database.cursor()
		cur.execute("SELECT * FROM dataset WHERE id=%s", (id,))
		rv = cur.fetchone()
		cur.close()
		data['select'] = rv
		return render_template('admin/edit_dataset.html', text=data)
	else :
		return redirect(url_for('login_admin'))

@app.route('/update_dataset', methods=['POST'])
def update_dataset():
	if request.method == 'POST':
			comment = request.form['comment']
			label = request.form['label']
			id = request.form['id']

			sql = "UPDATE dataset SET comment=%s, label=%s WHERE id=%s"
			data = (comment, label, id, )
			cur = database.cursor()
			cur.execute(sql, data)
			database.commit()
			flash('Data Dataset Berhasil Diperbarui !')
			return redirect(url_for('data_dataset_admin'))

@app.route('/hapus_dataset/<string:id>', methods=["GET"])
def hapus_dataset(id):
    cur = database.cursor()
    cur.execute("DELETE FROM dataset WHERE id=%s", (id,))
    database.commit()
    return redirect(url_for('data_dataset_admin'))

@app.route('/data_latih')
def data_latih():
	if 'username' in session:
		data={}
		user = session['username']
		data['username'] = user

		cur = database.cursor()
		cur.execute("SELECT * FROM data_latih")
		rv = cur.fetchall()
		cur.close()
		data['select'] = rv
		return render_template('admin/data_latih.html', prepros=data)
	else :
		return redirect(url_for('login_admin'))

@app.route('/prepro')
def prepro():
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

	#stemming
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

	return redirect(url_for('data_latih'))

@app.route('/data_klasifikasi_admin')
def data_klasifikasi_admin():
	if 'username' in session:
		data={}
		user = session['username']
		data['username'] = user

		cur = database.cursor()
		cur.execute("SELECT * FROM data_uji")
		rv = cur.fetchall()
		cur.close()
		data['select'] = rv
		return render_template('admin/data_klasifikasi_admin.html', result=data)
	else :
		return redirect(url_for('login_admin'))

@app.route('/edit_hasil_klasifikasi/<string:id>', methods=['GET'])
def edit_hasil_klasifikasi(id):
	if 'username' in session:
		data={}
		user = session['username']
		data['username'] = user

		cur = database.cursor()
		cur.execute("SELECT * FROM data_uji WHERE id=%s", (id,))
		rv = cur.fetchone()
		cur.close()
		data['select'] = rv
		return render_template('admin/edit_hasil_klasifikasi.html', text=data)
	else :
		return redirect(url_for('login_admin'))

@app.route('/update_hasil_klasifikasi', methods=['POST'])
def update_hasil_klasifikasi():
	if request.method == 'POST':
			tweet = request.form['tweet']
			label = request.form['label']
			id = request.form['id']

			sql = "UPDATE data_uji SET tweet=%s, label=%s WHERE id=%s"
			data = (tweet, label, id, )
			cur = database.cursor()
			cur.execute(sql, data)
			database.commit()
			flash('Data Dataset Berhasil Diperbarui !')
			return redirect(url_for('data_klasifikasi_admin'))

@app.route('/hapus_klasifikasi/<string:id>', methods=["GET"])
def hapus_klasifikasi(id):
    cur = database.cursor()
    cur.execute("DELETE FROM data_uji WHERE id=%s", (id,))
    database.commit()
    return redirect(url_for('data_klasifikasi_admin'))

@app.route('/data_model_klasifikasi')
def data_model_klasifikasi():
	if 'username' in session:
		data={}
		user = session['username']
		data['username'] = user
		return render_template('admin/data_model_klasifikasi.html', model=data)
	else:
		return redirect(url_for('login_admin'))

@app.route('/model_klasifikasi', methods=['POST'])
def model_klasifikasi():
	if 'username' in session:
		data_pre = pan.read_sql('SELECT * FROM data_latih', con=database)

		'''ambil_pre = []
		for index, value in data_pre.iterrows():
			ambil_pre.append(word_tokenize(value["comment"]))'''

		tf = CountVectorizer()
		vec = TfidfVectorizer(smooth_idf=False,norm=None)
		train_matrix = vec.fit_transform(data_pre['comment'].values.astype('U')).toarray()
		transformed_data_latih = tf.fit_transform((data_pre['comment']))

		X_train = train_matrix
		y_train = data_pre['label_comment']

		ambil = request.form['pohon']
		pohon = int(ambil)

		rf = RandomForestClassifier(n_estimators= pohon, criterion = 'gini', max_features='sqrt')
		rf.fit(X_train,y_train)

		joblib_file = "C:/Users/Tulenesia/sara/randomforest.sav"
		joblib.dump(rf, joblib_file)
		return redirect(url_for('hasil_model'))
	else:
		return redirect(url_for('login_admin'))

@app.route('/hasil_model')
def hasil_model():
	if 'username' in session:
		data={}
		user = session['username']
		data['username'] = user

		joblib_file = "C:/Users/Tulenesia/sara/randomforest.sav"
		random = joblib.load(joblib_file)
		data['rf'] = random
		return render_template('admin/hasil_model.html', model=data)
	else:
		return redirect(url_for('login_admin'))

@app.route('/data_pengujian')
def data_pengujian():
	if 'username' in session:
		data={}
		user = session['username']
		data['username'] = user
		return render_template('admin/data_pengujian.html', text=data)

@app.route('/hasil_pengujian', methods=['POST'])
def hasil_pengujian():
	if 'username' in session:
		data={}
		user = session['username']

		data_train = pan.read_sql('SELECT * FROM data_latih', con=database)

		jml    = len(data_train)

		ambil1 = request.form['pohon']
		ambil = request.form['split']
		ambil2 = int(ambil)
		pohon  = int(ambil1)
		split  = float(ambil2/100)

		tf = CountVectorizer()
		vec = TfidfVectorizer(smooth_idf=False,norm=None)
		train_matrix = vec.fit_transform(data_train['comment'].values.astype('U')).toarray()

		X = train_matrix
		y = data_train['label_comment']

		from sklearn.model_selection import train_test_split
		X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=split,random_state=42)

		rf = RandomForestClassifier( n_estimators= pohon,criterion = 'gini',max_features='sqrt',max_depth=None)
		rf.fit(X_train,y_train)
		y_pred = rf.predict(X_test)

		latih = len(y_train)
		uji   = len(y_test)

		from sklearn.metrics import confusion_matrix
		cnf_matrix = confusion_matrix(y_test, y_pred)

		FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
		FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
		TP = np.diag(cnf_matrix)
		TN = cnf_matrix.sum() - (FP + FN + TP)
		pre = TP/(TP+FP)
		rec = TP/(TP+FN)
		aku = (TP+TN)/(TP+FP+FN+TN)
		presisi = sum(pre)/4
		recall  = sum(rec)/4
		akurasi = sum(aku)/4

		data['username'] = user
		data['jumlah']   = jml
		data['latih']    = latih
		data['uji']      = uji
		data['pohon']    = pohon
		data['split']    = split
		data['matrix']   = cnf_matrix
		data['precision'] = presisi
		data['recall']    = recall
		data['accuracy']  = akurasi
		return render_template('admin/hasil_pengujian.html', text=data)
	else:
		return redirect(url_for('login_admin'))

#===============================================<><><>=================================================================================

@app.route('/index_member')
def index_member():
	if 'username' in session:
		data={}
		user = session['username']
		data['username'] = user
		return render_template('member/index_member.html', text=data)
	else:
		return redirect(url_for('index'))

@app.route('/edit_profil')
def edit_profil():
	if 'username' in session:
		data={}
		user = session['username']
		data['username'] = user

		cur = database.cursor()
		cur.execute("SELECT * FROM member WHERE username=%s", (user,))
		rv = cur.fetchall()
		cur.close()
		data['select'] = rv
		return render_template('member/edit_profil.html', text=data)
	else:
		return redirect(url_for('index'))

@app.route('/update_profil/<string:username>', methods=['GET'])
def update_profil(username):
	if 'username' in session:
		data={}
		user = session['username']
		data['username'] = user

		cur = database.cursor()
		cur.execute("SELECT * FROM member WHERE username=%s", (user,))
		rv = cur.fetchone()
		cur.close()
		data['select'] = rv
		return render_template('member/update_profil.html', text=data)
	else :
		return redirect(url_for('index'))

@app.route('/simpan_profil', methods=['POST'])
def simpan_profil():
		if request.method == 'POST':
			nama = request.form['nama']
			no_hp = request.form['no_hp']
			username = request.form['username']
			password = request.form['password']

			cur = database.cursor()
			cur.execute("SELECT * FROM member WHERE username=%s OR no_hp=%s",(username,no_hp,))
			user = cur.fetchall()

			if len(user) == 0:
				sql = "UPDATE member SET username=%s, password=%s nama_lengkap=%s, no_hp=%s WHERE username=%s"
				data = (username, password, nama, no_hp,)
				cur.execute(sql, data)
				database.commit()
				return redirect(url_for('data_member'))
			else:
				return "USERNAME / NO HANDPHONE SUDAH DIGUNAKAN"
			return redirect(url_for('edit_profil'))

@app.route('/aduan_konten')
def aduan_konten():
	if 'username' in session:
		data={}
		user = session['username']
		data['username'] = user
		return render_template('member/aduan_konten.html', text=data)
	else:
		return redirect(url_for('index'))

UPLOAD_FOLDER = 'C:/Users/Tulenesia/sara/foto_aduan'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#pengecekan file degan menggunakan rsplit
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/simpan_input',methods=["GET","POST"])
def simpan_input():
	if 'username' in session:
		if request.method == 'POST':
			file = request.files['file']

			if 'file' not in request.files:
				return render_template('member/aduan_konten.html')

			if file.filename == '':
				return render_template('member/aduan_konten.html')

			if file and allowed_file(file.filename):
				filename = secure_filename(file.filename)
				file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

				url = request.form['input']

				html = urlopen(url)
				soup = BeautifulSoup(html, 'lxml')
				nama = soup.find('span', class_="FullNameGroup").text
				username_tweet = soup.find('span', class_="username u-dir u-textTruncate").text
				tweet = soup.find('p', class_="TweetTextSize TweetTextSize--jumbo js-tweet-text tweet-text").text
				waktu = soup.find('span', class_="metadata").text
				retweet = soup.find('a', class_="request-retweeted-popup")
				if retweet == None:
					retweet1 = 'null'
				else:
					retweet1 = retweet.text

				like = soup.find('a', class_="request-favorited-popup")
				if like == None:
					like1 = 'null'
				else:
					like1 = like.text

				link = str(url)
				comment = str(tweet)
				input_comment = comment.replace("[^a-zA-Z]+", " ")
				input_comment = input_comment.replace(" +", " ")
				input_comment = input_comment.strip('[123.!? \n\t]')
				inputan = input_comment.lower()

				tokens = word_tokenize(inputan)
				separator= ' '
				token = separator.join(tokens)

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
				for idx,value in enumerate (tokens):
				    if value in table:
				        spelling.append(table[value])
				    else:
				        spelling.append(value)
				spell = ' '.join(spelling)

				stop_factory = StopWordRemoverFactory()
				stopword_fac = stop_factory.create_stop_word_remover()
				stopword = stopword_fac.remove(spell)

				factory = StemmerFactory()
				stemmer = factory.create_stemmer()
				stemming = (stemmer.stem(stopword))
				hasil = [stemming]

				data_pre = pan.read_sql('SELECT * FROM data_latih', con=database)

				tf = CountVectorizer()
				ft = CountVectorizer()
				vec = TfidfVectorizer(smooth_idf=False,norm=None)

				train_matrix = vec.fit_transform(data_pre['comment'].values.astype('U')).toarray()
				test_matrix = vec.transform(hasil)
				transformed_data_latih = tf.fit_transform((data_pre['comment']))
				transformed_data_uji = tf.transform(hasil)
				ft_fitur = ft.fit_transform(hasil)

				fitur = vec.get_feature_names()
				tf_fitur = tf.get_feature_names()
				fitur_uji = ft.get_feature_names()

				freq_uji = np.ravel(test_matrix.sum(axis=0))
				freq_uji1 = np.ravel(ft_fitur.sum(axis=0))
				freq_latih = np.ravel(transformed_data_latih.sum(axis=0))
				data_latih = list(zip(tf_fitur, freq_latih))
				
				data_uji1 = list(zip(fitur_uji, freq_uji1))
				data_uji = list(zip(fitur, freq_uji))
				
				y_test = test_matrix
				joblib_file = "C:/Users/Tulenesia/sara/randomforest.sav"
				rf = joblib.load(joblib_file)
				aa = rf.predict(y_test)
				hasil_akhir = str(aa)

				hasiltext = ''
				label = ''
				if aa == [0]:
					hasiltext = '"Pesan TIDAK Termasuk Kategori SARA (NON_SARA)"'
					label = 'NON_SARA'
				elif aa == [1]:
					hasiltext = '"Pesan Termasuk Kategori SARA"'
					label = 'SARA'

				current = datetime.now()
				tahun   = current.year
				bulan   = current.month
				hari    = current.day
				tgl     = date(int(tahun),int(bulan),int(hari))

				data = {}
				user = session['username']
				data['username'] = user
				data['input']  = comment
				data['output'] = hasiltext
				data['random'] = rf
				data['prepro'] = hasil
				data['case']   = inputan
				data['tokens'] = tokens
				data['spelling'] = spell
				data['stopword'] = stopword
				data['stemming'] = stemming
				data['tf_idf']   = data_uji
				data['fitur']  = data_uji1	
				data['label']  = label				

				cur = database.cursor()
				cur.execute("INSERT INTO data_uji (username_member, url, nama, username_tweet, tweet, tgl_tweet, jml_retweet, jml_like, label, tgl_input) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(user,link,nama,username_tweet,comment,waktu,retweet1,like1,label,tgl,))
				database.commit()

				return render_template('member/simpan_input.html', masuk=data)

@app.route('/riwayat_aduan')
def riwayat_aduan():
	if 'username' in session:
		data = {}
		user = session['username']
		data['username'] = user
		
		cur = database.cursor()
		cur.execute("SELECT id,url,username_tweet,tweet,tgl_tweet,label,tgl_input FROM data_uji WHERE username_member=%s",(user,))
		rv = cur.fetchall()
		cur.close()
		data['select'] = rv
		return render_template('member/riwayat_aduan.html', text=data)
	else :
		return redirect(url_for('index'))

@app.route('/hapus_aduan/<string:id>', methods=["GET"])
def hapus_aduan(id):
    cur = database.cursor()
    cur.execute("DELETE FROM data_uji WHERE id=%s", (id,))
    database.commit()
    return redirect(url_for('riwayat_aduan'))

#===============================================<><><>=================================================================================

@app.route('/logout')
def logout():
    session.pop('username')
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)


'''
				poh = request.form['pohon']
				pohon = int(poh)
				estimator = rf.estimators_[pohon]
				export_graphviz(estimator, out_file='pohon_keputusan.dot', 
		                class_names = ['Irrelevant','Netral','Non Sara','Sara'],
		                rounded = True, proportion = False, 
		                precision = 2, filled = True)

				# Convert to png
				import pydot

				(graph,) = pydot.graph_from_dot_file('pohon_keputusan.dot')
				graph.write_png('pohon_keputusan.png')
				'''