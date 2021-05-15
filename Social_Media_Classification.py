#Note you have to do 3 changes in order to get the output
#1 Change the file name in the portion of test data in line 266
#2 Give number of clusters from the graph manually in line 731 by observing the point where the slope changes significantly
#3 Topics corresponding to each cluster will be seen in the console you just have to map it with the csv file 
#topic which pops in the first place is for file 0 and so on 

#importing useful libraries
import warnings
warnings.filterwarnings("ignore")  
import numpy as np
import pandas as pd
import nltk
import os
from nltk.corpus import stopwords
from io import StringIO
from nltk.tokenize import word_tokenize
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
%matplotlib inline
from nltk.cluster import KMeansClusterer
from sklearn.manifold import TSNE
from sklearn import cluster
from sklearn import metrics
import operator
nltk.download('wordnet')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



#Loading the training data the name here should not be changed when running the algorithm for new data as this is training data
with open('english_singapore.csv',encoding='utf-8',errors='ignore') as infile:
    data=infile.read()
from io import StringIO
tweets=pd.read_csv(StringIO(data))


###Selecting the Important features from the training data which are either directly useful or used to extract certain features

df=tweets.loc[:, ['content','fluency_level','sentiment','source_type',
 'reach',
 'source_extended_attributes.facebook_followers',
 'source_extended_attributes.twitter_followers',
 'source_extended_attributes.instagram_followers',
 'article_extended_attributes.instagram_likes',
 'article_extended_attributes.facebook_likes',
 'article_extended_attributes.twitter_likes',
 'extra_author_attributes.description',
 'extra_source_attributes.name',
 'extra_author_attributes.short_name','tags_internal']]

##LIKES AND COMMENTS SUMMATION(twitter,facebook,insta) AND RE-NAMING the columns

df=df.rename(columns = {'extra_source_attributes.name':'name'})
df=df.rename(columns = {'extra_author_attributes.short_name':'short_name'})
df=df.rename(columns = {'extra_author_attributes.description':'description'})

df['likes']=df['article_extended_attributes.instagram_likes']+df['article_extended_attributes.facebook_likes']+df['article_extended_attributes.twitter_likes']
df['followers']=df['source_extended_attributes.instagram_followers']+df['source_extended_attributes.twitter_followers']+df['source_extended_attributes.facebook_followers']

df=df.drop(columns=['article_extended_attributes.facebook_likes',
'article_extended_attributes.instagram_likes','article_extended_attributes.twitter_likes'])

df=df.drop(columns=['source_extended_attributes.facebook_followers',
'source_extended_attributes.twitter_followers','source_extended_attributes.instagram_followers'])



#######FINDING THE PRESENCE AND ABSENCE OF URL####### and the pattern is defined based on the training data
    
import re
import numpy as np

#pattern capturing the links like  bit.ly/2yd25, abcd.com except pic.twitter.com
pattern=re.compile(r'(https?://[^t.co])|(\w+\.\w\w/\w+\d+)|[^pic\.twitter\.com]\.com',re.IGNORECASE)
length=len(df)
df['url_p']=pd.Series(np.random.randn(length),index=df.index)
url=[]
for i in range(0,len(df)):
    url.append(bool(pattern.search(str(tweets['content_snippet'][i]))))
df['url_p']=url


########Removing the urls from content########

url_remove=[]
for i in range(0,len(df)):
    url_remove.append(re.sub(r"(http\s?\S+)|(\.com$)|(www.\S+)"," ",str(df['content'][i])))
df['content']=url_remove 

###Finding the presence of phone number and address with some defined pattern as observed from training data

pattern=re.compile('([0-9]+?/[a-z]+)|[0-9]+[\-,\s][0-9]+|(wh?a?t\'?s\s?app)|([a-zA-Z]\s[^(=\\\\#)]\d\d[\d,\s][\d\s,][\d\s\-,][\d][\d\s,][\d][\d\s-]*)',re.IGNORECASE)
length=len(df)
df['phone_address']=pd.Series(np.random.randn(length),index=df.index)
phone_number=[]
for i in range(0,len(df)):
    phone_number.append(bool(pattern.search(str(df['content'][i]))))
df['phone_address']=phone_number

####Finding the presence and absence of keywords related to price

pattern=re.compile(r'\dkg|%|\$|AUD|HKD|INR',re.IGNORECASE)
length=len(df)
df['size_dollar']=pd.Series(np.random.randn(length),index=df.index)
size_dollar=[]
for i in range(0,len(df)):
    size_dollar.append(bool(pattern.search(str(df['content'][i]))))
df['size_dollar']=size_dollar

#######Calculating word_count in a post
df['Word_count']=df['content'].apply(lambda x:len(str(x).split(" ")))       

##Number of hastags in a post
df['hastags'] =df['content'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
        
##Number of tagging in a post
df['tagging'] =df['content'].apply(lambda x: len([x for x in x.split() if x.startswith('@')]))

##Lower Case
df['content'] =df['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))

##Remove Punctuation
df['content']= df['content'].str.replace('[^\w\s]',' ')

##Number Of stopwords
from nltk.corpus import stopwords
stop=stopwords.words('english')

##Removal of Stop_words
df['content'] = df['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#Removing all the words with length<=3
short_words=[]
for i in range(0,len(df)):
    short_words.append(re.sub(r'\W*\b\w{1,3}\b'," ",str(df['content'][i])))
df['content']=short_words


#######CONCATENATED WORDS TO MEANINGFUL SMALL WORDS
from math import log

# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).

words = open("words.txt").read().split()
wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
maxword = max(len(x) for x in words)

def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""
    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)
    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)
    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k
    return " ".join(reversed(out))


###################FINDING THE  presence of author_name and author_short_name in content
length=len(df)
df['name_present']=pd.Series(np.random.randn(length),index=df.index)

with open("english_words.txt") as word_file:
    english_words = set(word.strip().lower() for word in word_file)

def is_english_word(word):
    return word.lower() in english_words

def intersection(lst1,lst2):
    lst3=[value for value in lst1 if value in lst2]
    return lst3


name_present=[]

#if there is anything in intersection between name,short_name and the content extended with description then we have name_present

for i in range(0,len(df)):
   #Only Keeping the letters from A to Z
   Content1=re.sub('[^a-zA-Z]',' ',df['content'][i])
   Content1=Content1.lower()
   Content1=Content1.split()
   if(str(df['description'][i])!='nan'):
       Content3=re.sub('[^a-zA-Z]',' ',df['description'][i])
       Content3=Content3.lower()
       Content3=Content3.split()
   else:
       Content3=[] 
       
   Content3.extend(Content1)
   Content3=list(set(Content3))
   Content2=re.sub('[^a-zA-Z]',' ',str(df['short_name'][i]))
   Content2=Content2.lower()
   Content2=Content2.split()
   
   Content4=re.sub('[^a-zA-Z]',' ',str(df['name'][i]))
   Content4=Content4.lower()
   Content4=Content4.split()
   Content4=[infer_spaces(word) for word in Content4]
   Content4=[word for word in Content4 if is_english_word(word)]
   Content2.extend(Content4)
   
   Content2=[word for word in Content2 if len(word)>3]
   lst3=intersection(Content3,Content2)
   if(len(lst3)>0):
       name_present.append(True)
   else:
       name_present.append(False)
df['name_present']=name_present

###########Finding the presence of contact details in the content

df1=df.iloc[:,:]
contact_list=['ig','fb','dm','ltd','msg','email','text','call','mobile','inbox','link','check','contact']
df1['contact']=pd.Series(np.random.randn(len(df1)),index=df1.index)
contact=[]   
for i in range(0,len(df1)):
    cont=re.sub('[^a-zA-Z]',' ',df1['content'][i])
    cont=cont.lower()
    cont=cont.split()
    if(len(intersection(cont,contact_list))>0):
        contact.append(True)
    else:
        contact.append(False)
df1['contact']=contact
df['contact']=df1['contact']


#label encoding the categorical variable and removing the variables which are not that important to the model

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['url_p'] = labelencoder.fit_transform(df['url_p'])
df['phone_address'] = labelencoder.fit_transform(df['phone_address'])
df['size_dollar'] = labelencoder.fit_transform(df['size_dollar'])
df['contact'] = labelencoder.fit_transform(df['contact'])
df['name_present'] = labelencoder.fit_transform(df['name_present'])
df=df.drop(['source_type','tags_internal','reach','fluency_level','short_name','name','description'],axis=1)

#Loading the test data and performing the same set of analysis here
#the name singapore_labelled.csv should be changed according to the name of test data
#All the files saved at the same location would be better 

import pandas as pd
with open('singapore_labelled.csv',encoding='utf-8',errors='ignore') as infile:
    data=infile.read()
from io import StringIO
tweets1=pd.read_csv(StringIO(data))

###Selecting the Important features
dataset=tweets1.loc[:, ['content','fluency_level','sentiment','source_type',
 'reach',
 'source_extended_attributes.facebook_followers',
 'source_extended_attributes.twitter_followers',
 'source_extended_attributes.instagram_followers',
 'article_extended_attributes.instagram_likes',
 'article_extended_attributes.facebook_likes',
 'article_extended_attributes.twitter_likes',
 'extra_author_attributes.description',
 'extra_source_attributes.name',
 'extra_author_attributes.short_name','tags_internal']]

##LIKES AND COMMENTS SUMMATION(twitter,facebook,insta) AND RE-NAMING

dataset=dataset.rename(columns = {'extra_source_attributes.name':'name'})
dataset=dataset.rename(columns = {'extra_author_attributes.short_name':'short_name'})
dataset=dataset.rename(columns = {'extra_author_attributes.description':'description'})

dataset['likes']=dataset['article_extended_attributes.instagram_likes']+dataset['article_extended_attributes.facebook_likes']+dataset['article_extended_attributes.twitter_likes']
dataset['followers']=dataset['source_extended_attributes.instagram_followers']+dataset['source_extended_attributes.twitter_followers']+dataset['source_extended_attributes.facebook_followers']

dataset=dataset.drop(columns=['article_extended_attributes.facebook_likes',
'article_extended_attributes.instagram_likes','article_extended_attributes.twitter_likes'])

dataset=dataset.drop(columns=['source_extended_attributes.facebook_followers',
'source_extended_attributes.twitter_followers','source_extended_attributes.instagram_followers'])


#######FINDING THE PRESENCE AND ABSENCE OF URL#######
import re
import numpy as np
pattern=re.compile(r'(https?://[^t.co])|(\w+\.\w\w/\w+\d+)|[^pic\.twitter\.com]\.com',re.IGNORECASE)
length=len(dataset)
dataset['url_p']=pd.Series(np.random.randn(length),index=dataset.index)
url_p=[]
for i in range(0,len(dataset)):
    url_p.append(bool(pattern.search(str(tweets1['content_snippet'][i]))))
dataset['url_p']=url_p

########REMOVING THE URL########
url_remove=[]
for i in range(0,len(dataset)):
    url_remove.append(re.sub(r"(http\s?\S+)|(\.com$)|(www.\S+)"," ",str(dataset['content'][i])))
dataset['content']=url_remove 

###FINDING THE  presence of Phone number and address
pattern=re.compile('([0-9]+?/[a-z]+)|[0-9]+[\-,\s][0-9]+|(wh?a?t\'?s\s?app)|([a-zA-Z]\s[^(=\\\\#)]\d\d[\d,\s][\d\s,][\d\s\-,][\d][\d\s,][\d][\d\s-]*)',re.IGNORECASE)
length=len(dataset)
dataset['phone_address']=pd.Series(np.random.randn(length),index=dataset.index)
phone_number=[]
for i in range(0,len(dataset)):
    phone_number.append(bool(pattern.search(str(dataset['content'][i]))))
dataset['phone_address']=phone_number

####FINDING THE presence of PRICE AND SIZE

pattern=re.compile(r'\dkg|%|\$|AUD|HKD|INR',re.IGNORECASE)
length=len(dataset)
dataset['size_dollar']=pd.Series(np.random.randn(length),index=dataset.index)
size_dollar=[]
for i in range(0,len(dataset)):
    size_dollar.append(bool(pattern.search(str(dataset['content'][i]))))
dataset['size_dollar']=size_dollar


#######Calculating word_count
dataset['Word_count']=dataset['content'].apply(lambda x:len(str(x).split(" ")))       

##Number of hastags
dataset['hastags'] =dataset['content'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
        
##Number of tagging
dataset['tagging'] =dataset['content'].apply(lambda x: len([x for x in x.split() if x.startswith('@')]))

##Lower Case
dataset['content'] =dataset['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))

##Remove Punctuation
dataset['content']= dataset['content'].str.replace('[^\w\s]',' ')

##Number Of stopwords
from nltk.corpus import stopwords
stop=stopwords.words('english')

##Removal of Stop_words
dataset['content'] = dataset['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

short_words=[]
for i in range(0,len(dataset)):
    short_words.append(re.sub(r'\W*\b\w{1,3}\b'," ",str(dataset['content'][i])))
dataset['content']=short_words

#######CONCATENATED WORDS TO MEANINGFUL SMALL WORDS
from math import log

# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
words = open("words.txt").read().split()
wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
maxword = max(len(x) for x in words)

def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""
    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)
    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)
    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k
    return " ".join(reversed(out))

###################FINDING THE presence of author_name and author_short_name in content
    
length=len(dataset)
dataset['name_present']=pd.Series(np.random.randn(length),index=dataset.index)

with open("english_words.txt") as word_file:
    english_words = set(word.strip().lower() for word in word_file)

def is_english_word(word):
    return word.lower() in english_words

def intersection(lst1,lst2):
    lst3=[value for value in lst1 if value in lst2]
    return lst3

corpus=[]
name_present=[]

for i in range(0,len(dataset)):
   #Only Keeping the letters from A to Z
   Content1=re.sub('[^a-zA-Z]',' ',dataset['content'][i])
   Content1=Content1.lower()
   Content1=Content1.split()
   if(str(dataset['description'][i])!='nan'):
       Content3=re.sub('[^a-zA-Z]',' ',dataset['description'][i])
       Content3=Content3.lower()
       Content3=Content3.split()
   else:
       Content3=[] 
   Content3.extend(Content1)
   Content3=list(set(Content3))
   Content2=re.sub('[^a-zA-Z]',' ',str(dataset['short_name'][i]))
   Content2=Content2.lower()
   Content2=Content2.split()
   Content4=re.sub('[^a-zA-Z]',' ',str(dataset['name'][i]))
   Content4=Content4.lower()
   Content4=Content4.split()
   Content4=[infer_spaces(word) for word in Content4]
   Content4=[word for word in Content4 if is_english_word(word)]
   Content2.extend(Content4)
   Content2=[word for word in Content2 if len(word)>3]
   lst3=intersection(Content3,Content2)
   if(len(lst3)>0):
       name_present.append(True)
   else:
       name_present.append(False)
dataset['name_present']=name_present

###########Finding the contact details, whether contact information is present or not
dataset1=dataset.iloc[:,:]
contact_list=['ig','fb','dm','ltd','msg','email','text','mobile','call','inbox','link','check','contact']
dataset1['contact']=pd.Series(np.random.randn(len(dataset1)),index=dataset1.index)
contact=[]   
for i in range(0,len(dataset1)):
    cont=re.sub('[^a-zA-Z]',' ',dataset1['content'][i])
    cont=cont.split()
    if(len(intersection(cont,contact_list))>0):
        contact.append(True)
    else:
        contact.append(False)
dataset1['contact']=contact
dataset['contact']=dataset1['contact']


#label encoding required varibles and dropping some variable

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
dataset['url_p'] = labelencoder.fit_transform(dataset['url_p'])
dataset['phone_address'] = labelencoder.fit_transform(dataset['phone_address'])
dataset['size_dollar'] = labelencoder.fit_transform(dataset['size_dollar'])
dataset['contact'] = labelencoder.fit_transform(dataset['contact'])
dataset['name_present'] = labelencoder.fit_transform(dataset['name_present'])
dataset=dataset.drop(['source_type','tags_internal','reach','fluency_level','short_name','name','description'],axis=1)


#Concatinating the test and the train data inorder to capture the context of both training and test set in the word vectors
frames = [df, dataset]
data = pd.concat(frames)
data=data.reset_index()
data = data.iloc[:,1:]

# getting tfidf values for the concatanated data
import pickle
loaded_vec=TfidfVectorizer(input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, analyzer='word', stop_words = None, token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), max_df=.5, min_df=30, max_features=200, vocabulary=None, binary=False,  norm='l2',use_idf=True, smooth_idf=True, sublinear_tf=False)
X_test_tfidf= loaded_vec.fit_transform(data['content'])
X_test_tfidf =pd.DataFrame(X_test_tfidf.toarray())
X_test_tfidf.columns=loaded_vec.vocabulary_


#lemmatizing the words in the content to their base word
import nltk
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
data['content'] = data.content.apply(lemmatize_text)

# Converting the glove pretrained  model into word2vec and then training the pretrained model on our data

from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_input_file="clustering.txt", word2vec_output_file="word2vec.txt")
from gensim.models.keyedvectors import KeyedVectors
glove_model = KeyedVectors.load_word2vec_format("word2vec.txt", binary=False)

#Traing the model on our data
model_2 = gensim.models.Word2Vec (data['content'],min_count=1,size=200,window=5,negative=20,seed=1000)
total_examples = model_2.corpus_count

#Updating the model on our data based on the pretrained model
model_2.build_vocab([list(glove_model.vocab.keys())], update=True)
model_2.intersect_word2vec_format("word2vec.txt", binary=False, lockf=1.0)
model_2.train(data['content'], total_examples=total_examples, epochs=model_2.iter)

# Getting sentence vector through weighted average of word vector and tf idf score 
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

# Basically a vocabulary of the words in our dataset
dct = Dictionary(data['content']) 
 
#Creating corpus for every tweet
corpus = [dct.doc2bow(line) for line in data['content']]  # convert corpus to BoW format

#Fitting the tfidf model on the corpus
model_tfidf = TfidfModel(corpus)

#Initializing an empty list for sentence vector 
sent_vec=[]

#Sent_vec=sum over length of sentence(Tf_idf*Word vector)/Sum tf_idf
for i in range(len(data['content'])):
    weighted = np.zeros(200)
    sum_tfidf=0
    for j in range(len(list(set(data['content'][i])))):
        weighted= (weighted)+ (model_tfidf[corpus[i]][j][1])* (model_2[list(set(data['content'][i]))[j]])
        sum_tfidf+=model_tfidf[corpus[i]][j][1]
    sent_vec.append((weighted)/sum_tfidf)



#If in case the above code gives error of unavailability of a particular word, use this part instead of using from line 511-534
## Function for taking the average of word vectors
#def sent_vectorizer(sent, model):
#    sent_vec =[]
#    numw = 0
#    for w in sent:
#        try:
#            if numw == 0:
#                sent_vec = model[w]
#            else:
#                sent_vec = np.add(sent_vec, model[w])
#            numw+=1
#        except:
#            pass
#    
#    return np.asarray(sent_vec) / numw
#
#
## getting the word vector
#sent_vec=[]
#for sentence in data['content']:
#    sent_vec.append(sent_vectorizer(sentence, model_2))


#Getting a dataframe of our sent vec 
ln = len(sent_vec[0])
 # Initializing columns in dictionary to capture all dimensions of vector
rv = {}
#Initializing Dimensions 
  
for i in range(ln):
    st = "dim_"+ str(i+1)
    rv[st] = []
    
##----------------- No. of columns = Dimensions in tweet vector vector----------------------- 
for k in range(len(sent_vec)):
    for i in range(ln):
        st = "dim_"+ str(i+1)
        rv[st].append(sent_vec[k][i])
      

    if(k%100) == 0 :
      print(k," rows done")
dim2 = pd.DataFrame(data = rv)

#concatinating sent_vec, tfidf_vector and the features extracted from the text
new_data = pd.concat([data,dim2,X_test_tfidf],axis=1)

#Performing operations on the new_data to get our final training and test data

data_train= new_data.iloc[:len(df),:] 
data_test=new_data.iloc[len(df):,:]
data_train=data_train.drop(['content','followers','likes','sentiment','tagging'],axis=1)
data_train['label']=tweets['label']

data_test=data_test.reset_index()
data_test['label']=tweets1['label']
data_test=data_test.drop(['followers'],axis=1)
data_test=data_test.drop(['likes','sentiment','tagging'],axis=1)
data_test=data_test.drop(['index'],axis=1)

data_testdf=data_test.iloc[:,:]
data_test=data_test.drop(['content'],axis=1)
data_test=data_test.drop(['label'],axis=1)


# splitting the data for training and test

X=data_train.drop(['label'],axis=1)
Y= data_train['label']



#Training the xgboost classifier with its tuned hyperparameter


from sklearn.model_selection import GridSearchCV
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2),
 'reg_alpha':range(0,1,10)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X,Y)

#predicting the output on test data
predicted=gsearch1.predict(data_test)





# Separating Influencers from total posts using the rule defined 
pattern=re.compile(r'(\w?\w?blogger)|(\w?\w?blogged)',re.IGNORECASE)
for i in range(len(data_testdf)):
    if bool(pattern.search(str(data_testdf['content'][i])))==True:
        predicted[i]=0
        
###########Finding the tweets from consumer which are influencer by defining certain rules
data_testdf['followers']=tweets1['source_extended_attributes.instagram_followers']+tweets1['source_extended_attributes.twitter_followers']+tweets1['source_extended_attributes.facebook_followers']
influence_list=['coupon','promocode','discountcode','% off','deal','checkitout','contactme','check out','discount','buynow','worthit','ordernow','code','discount','promo','off','grab']
data_testdf['influ']=pd.Series(np.random.randn(len(data_testdf)),index=data_testdf.index)
influ=[]   

#Finding the length of intersection of the list above and tha actual content
for i in range(0,len(data_testdf)):
    cont=re.sub('[^a-zA-Z]',' ',str(data_testdf['content'][i]))
    cont=cont.lower()
    cont=cont.split()
    if(len(intersection(cont,influence_list))>0):
        influ.append(1)
    else:
        influ.append(0)

data_testdf['influ']=influ

#defining a rule for influencer

for i in range(len(data_testdf)):
    if predicted[i]==1:
        if (data_testdf['followers'][i]>5000) or (data_testdf['followers'][i]>1000 and data_testdf['influ'][i]==1):
           predicted[i]=0

data_testdf['predicted']=predicted
tweets1['predicted']=predicted

accuracy_score(tweets1['label'],tweets1['predicted'])
confusion_matrix(tweets1['label'],tweets1['predicted'])


tweets1.to_csv('marketer_consumer segregated.csv')


#Dimensionality reduction through pca
#converting 200 dimensional sentence vector to 30 dimensions
from sklearn.decomposition import PCA
    
pca = PCA(n_components=30)
pca.fit(dim2)
print(pca.explained_variance_ratio_) 
existing_2d = pca.transform(dim2)
existing_df_2d = pd.DataFrame(existing_2d)
existing_df_2d.index = dim2.index



#taking only sentence vector part of test data
dim2_clus=existing_df_2d.iloc[len(df):,:]
dim2_clus=dim2_clus.reset_index()
content=[]
url_c=[]

for i in range(len(tweets1)):
    content.append(tweets1['content'][i])
    url_c.append(tweets1['url'][i])
dim2_clus['content']=content
dim2_clus['url']=url_c

#taking only consumer's data
dim2_clus=dim2_clus[tweets1['predicted']==1]  
dim2_clus=dim2_clus.reset_index() 

#final data frame for clustering purpose
final_clus=dim2_clus.drop(['level_0','content','index','url'],axis=1)

#getting the metric for number of cluster selection using the elbow point method where kmeans and silhouette score is calculated for each number of cluster and a point where sudden change of slope is obtained is the point corresponding to optimal number of clusters
silhouette = {}
kmeans_score={}

for i in range(2,20):
    kmeans = cluster.KMeans(n_clusters=i,max_iter=500,tol=.0001)
    kmeans.fit(final_clus)
    labels = kmeans.labels_ 
    silhouette[i] = (metrics.silhouette_score(final_clus, labels, metric='euclidean'))
    kmeans_score[i] = -kmeans.score(final_clus)  


#Getting the maximum silhouette and kmeans score along with their index of number of clusters
maximum_silhouette = max(silhouette, key=silhouette.get) 
minimum_kmeans=min(kmeans_score,key=kmeans_score.get)
print(maximum_silhouette, silhouette[maximum_silhouette])
print(minimum_kmeans, kmeans_score[minimum_kmeans])  

#Plot between Number of clusters and respective kmeans score
lists1 = sorted(kmeans_score.items())
p, q = zip(*lists1)
plt.plot(p, q)


#Deciding the Number of clusters based on the above plot and clustering
NUM_CLUSTERS=4
kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS,max_iter=500,tol=.0002)
kmeans.fit(final_clus) 

#Getting labels 
labels = kmeans.labels_



#Getting count of posts in a cluster
unique, counts = np.unique(labels, return_counts=True)
dict(zip(unique, counts))

#Finally saving the  file with tweets and labels
dim2_clus=dim2_clus[['content','url']]
dim2_clus['labels']=labels
for i in range(NUM_CLUSTERS):
    tweets0=dim2_clus[dim2_clus['labels'] == i]
    tweets0.to_csv("choc"+str(i)+".csv")


#Getting topic from each of the cluster
from gensim import corpora, models
from pprint import pprint

alltop = []
for i in range(NUM_CLUSTERS):
    tweets_lda=dim2_clus[dim2_clus['labels']==i]
    tweets_lda=tweets_lda.reset_index()
    ##Lower Case
    tweets_lda['content'] =tweets_lda['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))

    ##Remove Punctuation
    tweets_lda['content']= tweets_lda['content'].str.replace('[^\w\s]',' ')

    ##Number Of stopwords
    from nltk.corpus import stopwords
    stop=stopwords.words('english')

    ##Removal of Stop_words
    tweets_lda['content'] = tweets_lda['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    short_words=[]
    for j in range(0,len(tweets_lda)):
        short_words.append(re.sub(r'\W*\b\w{1,3}\b'," ",str(tweets_lda['content'][j])))
    tweets_lda['content']=short_words
    tweets_lda['content'] = [word_tokenize(k) for k in tweets_lda['content']]

    dictionary = gensim.corpora.Dictionary(tweets_lda['content'])
    bow_corpus = [dictionary.doc2bow(doc) for doc in tweets_lda['content']]
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    for doc in corpus_tfidf:
        pprint(doc)
        break
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=1, id2word=dictionary, passes=2, workers=4,eval_every=15)
    fd = {}
    for idx, topic in lda_model_tfidf.print_topics(-1):
        fd[i] = topic
        print('Topic: {} Word: {}'.format(idx, topic))
    alltop.append(fd)   
    
topics=pd.DataFrame(alltop)   
topics.to_csv("topics.csv") 