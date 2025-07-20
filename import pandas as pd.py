import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import re

import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

plt.style.use('Solarize_Light2')
%matplotlib inline
df = pd.read_csv('../input/traindata/train.csv') # loading data
df.head() # looking at first five rows of the data
df.shape 
df.isnull().sum() 
df.dropna(inplace = True)
df.isna().sum()
df.shape
plt.figure(figsize = (9, 5))
sns.countplot(df['label'])

plt.show()
df.dtypes 
df['label'] = df['label'].astype(str)
df.dtypes
df.head(10)
df.reset_index(inplace = True) # resetting the index of data 
df.head(10)
df.drop(['index','id'],axis=1,inplace=True) # dropping 'index' and 'id' columns
df.head()
ps = PorterStemmer() # initializing porter stemmer
corpus=[]
sentences=[]
for i in range(0,len(df)):
    review=re.sub('[^a-zA-Z]',' ', df['title'][i])
    review=review.lower()
    list=review.split()
    review=[ps.stem(word) for word in list if word not in set(stopwords.words('english'))]
    sentences=' '.join(review)
    corpus.append(sentences)
corpus[0]
corpus[:20]
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 5000, ngram_range = (1, 3))
# splitting dataset into features and label 

X = cv.fit_transform(corpus).toarray()
y = df['label']
X, y
cv.get_feature_names()[0:20]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(X_train, y_train)
from sklearn import metrics

metrics.accuracy_score(y_test, y_pred)
def plot_confusion_matrix(cm):
    plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title('Confusion Matrix')
    labels=['positive','negative']
    tick_marks=np.arange(len(labels))
    plt.xticks(tick_marks,labels)
    plt.yticks(tick_marks,labels)
    cm = metrics.confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm)
from sklearn.linear_model import PassiveAggressiveClassifier

linear_clf = PassiveAggressiveClassifier()
linear_clf.fit(X_train, y_train)
y_pred = linear_clf.predict(X_test)
metrics.accuracy_score(y_test,y_pred)
cm2 = metrics.confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cm2)

feature_names = cv.get_feature_names()
sorted(zip(classifier.coef_[0],feature_names),reverse=True)[0:20]

feature_names = cv.get_feature_names()
sorted(zip(classifier.coef_[0],feature_names),reverse=True)[-20:]