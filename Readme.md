Petunjuk penggunaan:

POS-Tag-Classifier.py :
	- Merupakan program untuk eksperimen mencari tau kombinasi classifier dan dataset terbaik.
	- Dataset yang digunakan merupakan gabungan id-ud-train.conllu sebagai training set dan id-ud-dev.conllu sebagai testing set.
	- Untuk mempermudah eksperimen kami menggunakan file configuration.
	- python POS-Tag-Classifier.py <configuration file>

POS-Tag-Classify-Input-User.py :
 	- Merupakan program untuk memberikan POS Tag pada tiap kata pada suatu kalimat yang diinput oleh user.
 	- Model Classifier yang digunakan pada program ini merupakan model terbaik hasil eksperimen pada POS-Tag-Classifier.py
 	- python POS-Tag-Classify-Input-User.py