from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
import re
from nltk.stem.snowball import FrenchStemmer
import spacy 
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


nlp = spacy.load('fr_core_news_sm')
stemmer = FrenchStemmer()
print("*******************************************************************")
print ("downloading nltk ressources")
print("*******************************************************************")
nltk.download('stopwords')
nltk.download('punkt')
print("*******************************************************************")
print ("download finish")
print("*******************************************************************")

ps = PorterStemmer()
source = 'fake.txt'
# import french stop words list
stop_words = set(stopwords.words('french'))

f = open(source,'r')
appendFile = open('tweetPropre.txt','a') 

# create CountVectorizer object
vectorizer = CountVectorizer()
corpus = [
          'cache derrière Gorafi , humour dénoncer , être gentil Yann Barthès ?',
          'Paris – Lancement nouveau service livraison domicile mâche repas',
          '@pabl0mira avoir marre répondre.',
          ' @le_gorafi Écrêtage ? distraction ? tétracapillosectomie ?',
]
X = vectorizer.fit_transform(corpus)# columns of X correspond to the result of this method
vectorizer.get_feature_names() == (
    ['Gorafi', 'humour', 'avoir', 'distraction', 'tétracapillosectomie',
     'domicile', 'repas', 'dénoncer', 'Lancement', 'text',
     'Écrêtage', 'Paris', 'Barthès'])# retrieving the matrix in the numpy form
X.toarray()# transforming a new document according to learn vocabulary
# learn the vocabulary and store CountVectorizer sparse matrix in X
a = vectorizer.transform(['Barthès va à la plange et Gorafi humour.']).toarray()
newrow = (vectorizer.transform(['Barthès repas à la Paris tétracapillosectomie Gorafi humour.']).toarray())
a = np.vstack([a, newrow])
#inputVector.write(vectorizer.transform([token]).toarray())
np.savetxt("inputNeurone.csv", a, delimiter=",",fmt='%d')




# Remove stopwords
for line in f: 
       line = line.replace("«","")
       line = line.replace("»","")
       line = line.replace(",","")
       line = line.replace('"',"")
       line_split = line.split()
       new_line = ""
       for word in line_split:
              word = re.sub(r'^https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE)      
              new_line = new_line +" "+ word  
              corpus.append(word)
       doc = nlp(new_line)
      
       for token in doc:
              #print(token, token.lemma_) 
              #lemmatisation
              token = token.lemma_
              if token not in stop_words:
                     #print(token)
                     appendFile.writelines(" "+token)
                  
                     #print (word_tokenize(token)) 
       appendFile.writelines('\n')               
appendFile.close()
appendFile = open("tweetPropre.txt",'r')




f.close()
print("*******************************************************************")
print ("finish")
print("*******************************************************************")
