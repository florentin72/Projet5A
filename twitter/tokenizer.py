from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
import re
from nltk.stem.snowball import FrenchStemmer
import spacy 


nlp = spacy.load('fr_core_news_sm')



stemmer = FrenchStemmer()


nltk.download('stopwords')
print ("download finish")
ps = PorterStemmer()
source = 'fake.txt'
# import french stop words list
stop_words = set(stopwords.words('french'))


f = open(source,'r')


appendFile = open('filteredtext.txt','a') 
# Remove stopwords
for line in f: 
       line_split = line.split()
       for word in line_split:
              word = re.sub(r'^https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE)      
                     

       doc = nlp(line_split)
       for token in doc:
              print(token, token.lemma_) 
              #lemmatisation
              token = token.lemma_
              if token not in stop_words:
                     print(token)
                     appendFile.writelines(" "+token) 
              

       
       
         
        
appendFile.close()
       
f.close()
print("#########################################################################")

print ("finish")