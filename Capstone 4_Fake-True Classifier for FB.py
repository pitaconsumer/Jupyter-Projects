#!/usr/bin/env python
# coding: utf-8

# # Applying Fake-Authentic Classifier Over Facebook Political Ads

# ## Context:

# ## Problem:
# 
# ## Question: 

# ## Methodology:
# 

# ## Part 1: Learning a Classifier Model from Articles Review

# In[1]:


get_ipython().system("wget -c 'https://github.com/pitaconsumer/some-datasets/blob/master/572515_1037534_compressed_Fake.csv.zip?raw=true'")
get_ipython().system('wget -c "https://github.com/pitaconsumer/some-datasets/blob/master/572515_1037534_compressed_True.csv.zip?raw=true"')
get_ipython().system('unzip -o  "572515_1037534_compressed_Fake.csv.zip?raw=true"')
get_ipython().system('unzip -o "572515_1037534_compressed_True.csv.zip?raw=true"')


# In[ ]:


get_ipython().system('pip install spacy tqdm')


# In[ ]:


get_ipython().system('python -m spacy download en_core_web_lg')


# In[ ]:


get_ipython().system('pip install textblob')


# In[ ]:


get_ipython().system('pip install preprocessor')


# In[ ]:


get_ipython().system('pip install seaborn')


# In[116]:


get_ipython().system('pip install wordcloud')


# In[2]:


import spacy 
import en_core_web_lg

spacy.prefer_gpu()
nlp = en_core_web_lg.load()


# In[3]:


import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress bar")


import preprocessor
from textblob import TextBlob
import statistics
from typing import List


import scipy

import spacy
import nltk
import re

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer


# In[5]:


fake_df = pd.read_csv('Fake.csv')


# In[5]:


get_ipython().run_line_magic('time', 'fake_df.shape')


# In[6]:


true_df = pd.read_csv("True.csv")


# In[7]:


fake_vectors = fake_df['text'].progress_apply(lambda x: pd.Series(nlp(x).doc.vector.tolist()))


# In[8]:


true_vectors = true_df['text'].progress_apply(lambda x: pd.Series(nlp(x).doc.vector.tolist()))


# In[9]:


fake_vectors['y'] = 0
true_vectors['y'] = 1


# In[20]:


fake_vectors.to_pickle('fake_vectors.pickle')
true_vectors.to_pickle('true_vectors.pickle')


# In[10]:


true_df['y'] = 1
fake_df['y'] = 0

all_df= pd.concat([true_df, fake_df], ignore_index=True)


# In[22]:


all_df


# ## Testing Bag of Words Method
# We will compare and contrast results of using Random Forest Classifier with BOW method 

# In[ ]:


#Do I open pick from here?


# In[14]:


#Need to import library for Stop Words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[15]:


#Tokenize text for real news 

def clean_text(text):
    text = text.lower()
    #pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    #token = word_tokenize(text)
    words = [word for word in text.split(" ") if word.isalpha()]
    stop_words = set(stopwords.words("english"))
    stop_words.discard("not")
    PS = PorterStemmer()
    words = [PS.stem(w) for w in words if w not in stop_words]
    words = ' '.join(words)
    return words


# In[16]:


print(all_df['text'].head(1).apply(clean_text))


# In[17]:


# Let's take a look at the updated text
all_df['updated_text'] = all_df['text'].apply(clean_text)


# In[67]:


all_df[['updated_text', 'text']]


# In[18]:


from collections import Counter

c = Counter()

ignore_this = all_df['updated_text'].apply(lambda row: c.update(row.split(" ")))


# In[19]:


top_words = c.most_common(50)
top_words


# In[20]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
bag_of_words_fake_real = vectorizer.fit_transform(all_df['updated_text'])


# In[81]:


len(vectorizer.get_feature_names())


# In[82]:


from sklearn.model_selection import train_test_split


# In[108]:


bag_of_words_fake_real.shape


# In[86]:


all_df['y'].shape


# In[92]:


X_train, X_test, y_train, y_test = train_test_split(bag_of_words_fake_real, all_df['y'],
                                                    random_state=42,
                                                    test_size=0.33)


# In[95]:


y_train.shape


# In[97]:


from sklearn.ensemble import RandomForestClassifier

rfc_bag = RandomForestClassifier()

rfc_bag.fit(X_train, y_train)


# In[98]:


rfc_bag.score(X_train, y_train)


# In[99]:


rfc_bag.score(X_test, y_test)


# ### Word Cloud of Top 50 Words

# In[119]:


get_ipython().run_line_magic('pinfo', 'WordCloud')


# In[67]:


# most_common() produces k frequently encountered in: all_df['updated_text'](10) 
# input values and their respective counts. 
most_occur = Counter.most_common(top_words)

print(most_occur)


# ### Plot the Word Cloud

# In[69]:


from wordcloud import WordCloud
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
#% matplotlib inline

#c:\intelpython3\lib\site-packages\matplotlib\__init__.py:

import warnings
warnings.filterwarnings("ignore")

text_wc_visual = " ".join(text for text in all_df.updated_text)
# Create and generate a word cloud image: wordcloud = WordCloud().generate(text)

wordcloud_BOW = WordCloud().generate(text_wc_visual)
#wordcloud_BOW = WordCloud(width = 500, 
                      #contour_color = "purple",
                      #height= 300, 
                      #random_state = 21,
                      #max_words = 30,
                      #max_font_size =110).generate(top_words)
                    
plt.imshow(wordcloud_BOW, interpolation='bilinear')
plt.axis("off")
plt.show()                    


# ## METHOD TWO: NLTK
# 

# ## Testing Text Vector (spacy library) Method
# We will compare and contrast results of using Random Forest Classifier from BOW method with results from Text Vector method. 

# In[28]:


from nltk import TreebankWordTokenizer
tokenizer = TreebankTokenizer()
train['tokens'] = train['text'].map(tokenizer.tokens)


# In[100]:


all_vectors = pd.concat([fake_vectors, true_vectors], ignore_index=True)


# In[106]:


#Method 2 requires that X and y be trained as X2_, and y2_
X2_train, X2_test, y2_train, y2_test = train_test_split(all_vectors.drop(columns=['y']), 
                                                    all_vectors['y'], 
                                                    random_state=42,
                                                    test_size=0.33)


# In[107]:


#Run a random forest classifier on vectors
y2_train.shape


# In[109]:


#Classifier for Method 2 using vectors
rfc_vectors = RandomForestClassifier()

rfc_vectors.fit(X2_train, y2_train)


# In[111]:


#Apply classifier to X2 and y2 training set and obtain 'Score'.
rfc_vectors.score(X2_train, y2_train)


# In[110]:


#Apply classifier to X2 and y2 testing set and obtain 'Score'.
rfc_vectors.score(X2_test, y2_test)


# In[ ]:





# In[ ]:





# # Part 2: Facebook Political Ads Classified Into Fake Versus Authentic Via Random Forest Model 

# #### CLF from Training of fake data sets
# #### Predict outcomes for Facebook 
#     Y_fb_pred = clf.fit(X_train_fake, Y_train_true).predict(X_test_fb_pol)
# clf.fit(X_fb_pol, Y_fb_pred).  #Test Set is FB Dataset

# In[ ]:


get_ipython().system('pip install nltk')


# In[29]:


#from wordcloud import WordCloud,STOPWORDS
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import re,string,unicodedata
from nltk.stem import WordNetLemmatizer,PorterStemmer
import os
import gc
from nltk.tokenize import word_tokenize
from collections import  Counter

#stop = set(stopwords.words('english'))
#punctuation = list(string.punctuation)
#stop.update(punctuation)


# In[ ]:





# In[30]:


fb = pd.read_csv('fbpac-ads-en-US.csv.xz', dtype={'message': 'string', 'title': 'string', 'paid_for_by':'string'}) #'/Users/mehrunisaqayyum/Downloads/work/fbpac.csv'
fb


# In[17]:


fb.columns


# In[32]:


fb.dtypes


# In[31]:


fe = ['title','message','paid_for_by']
text_fb = fb[fe]
text_fb.head(-10)


# ### Text Cleaning
# We will review text in columns 'title','message','paid_for_by' for our Natural Language Processing project.

# In[32]:


'''Remove punctuation and "weird stuff like --" from ['title','message','paid_for_by'].'''

import re

def text_cleaner(text_fb):
    # Visual inspection identifies a form of punctuation spaCy does not
    # recognize: the double dash '--'.  Better get rid of it now!
    text1 = re.sub(r'--',' ',text_fb)
    text1 = re.sub("[\[].*?[\]]", "", text_fb)
    text1 = ' '.join(text_fb.split())
    return text1


# In[30]:


pd.__version__


# In[51]:


#need values to be strings 
text_fb.dtypes


# In[52]:


'''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
import re
import string

def clean_text_round1(text1):
    if pd.isna(text1):
        return text1
    
    text2 = text1.lower()
    text2 = re.sub('\[.*?\]', '', text1)
     #Add regex to address the 'p' where removing 'p' with brackets or remove first and last letter
    text2 = re.sub('\w*\d\w*', '', text1)
    text2 = re.sub('<.*?>', '', text1)
    text2 = re.sub('[%s]' % re.escape(string.punctuation), '', text1)
    return text2

# <.*?>


# In[53]:


clean_text_round1("HERE IS SOME text for you to clean, sorry about the CAPS LOCK.")


# In[54]:


#Test a 'message'
clean_text_round1(text_fb['message'].iloc[0])


# In[55]:


#Del suggestion to test clean
#Creating new column 'clean_message'

text_fb['clean_message'] = text_fb.message.apply(clean_text_round1)


# In[57]:


text_fb #See columns: ['message','clean_message']


# In[39]:


re.sub('<.*?>', '', text_fb['message'].iloc[0])


# In[63]:


get_ipython().run_line_magic('pinfo', 're.sub')


# In[58]:


#Dataset = text_fb
text_fb


# In[43]:


text_fb.title.str.lower()
#applied lower case method to title column
#how to iterate over every row?


# In[44]:


#Example: alice = text_cleaner(alice)
text_fb_nlp = clean_text_round1(text_fb)


# In[ ]:


text.dtypes


# #### Methodology Note: 
# We need to parse the cleaned data. The cleaned data represents the Facebook text in the 'message' column. We could include the 'title' column as well to match the 'title' with the corresponding message as a measure of checking authenticity the way the New York Times Challenge demonstrated.
# 
# Should we undertake: Topic Modeling?

# In[62]:


# Parse the cleaned ads. This can take a bit.
nlp = spacy.load('en_core_web_lg')
#fb_ad_doc = nlp(text_fb_nlp)  #doc = nlp("string")
fb_ad_doc = nlp(text_fb['clean_message'])  #trying to place data--the text-- from column

#Iterate over tokens in a doc
#for token in fb_ad_doc:
 #   print(token.text) #print (token.text)
    
    #Don't forget sentences_df


# ### Top Fifty Words
# 

# In[65]:


#Identify top 50 words in Facebook Messages 'text_fb'
f = Counter()

#ignore_this = all_df['updated_text'].apply(lambda row: c.update(row.split(" ")))
ignore_this2 = text_fb['clean_message'].apply(lambda row: f.update(row.split(" ")))


# In[66]:


top_words_fb = f.most_common(50)
top_words_fb


# In[ ]:





# In[ ]:





# ### WordCloud: Facebook Political Ad (Messages)
# Reviewing top 50 words in messages.

# In[71]:


# Create and generate a word cloud image: wordcloud = WordCloud().generate(text)
#text_wc_visual2 = " ".join(text for text in text_fb.clean_text)


wordcloud_BOW2 = WordCloud().generate(top_words_fb) #(text_wc_visual2)
#wordcloud_BOW = WordCloud(width = 500, 
                      #contour_color = "purple",
                      #height= 300, 
                      #random_state = 21,
                      #max_words = 30,
                      #max_font_size =110).generate(top_words)
                    
plt.imshow(wordcloud_BOW2, interpolation='bilinear')
plt.axis("off")
plt.show() 


# In[ ]:





# ### Importing second Facebook dataset to argue for business case.

# In[ ]:


fb_likes = pd.read_csv('/Users/mehrunisaqayyum/Downloads/pseudo_facebook.csv')
fb_likes


# In[ ]:


#Concatenating
fb_df2 = pd.concat([fb, fb_likes], ignore_index=True, sort =True)
fb_df2


# In[ ]:


fb_df2.columns


# In[ ]:


fb_df2.dtypes


# ### Feature Description: Second Facebook Dataset
# 
# ad_id is the id of specific ad set | Numeric
# 
# Reporting_start and reporting_end are the start and end dates of the each ad | Numeric
# 
# Campaign_id is the id assigned by the ad running company | Numeric- Negligible
# 
# fb_campaign_id is the id assigned by facebook for every ad set| Numeric- Negligible
# 
# age and gender talk about the demographics | Categorical
# 
# Interest1, Interest2, Interest3 are the user interests and likes of facebook users who were targeted for the ad | Categorical 
# 
# Impressions are the number of times the ad was shown to the users |Numeric
# 
# Clicks is the number of time users clicked on the ad | Numeric
# 
# spent is the amount of money spent on each campaign | Numeric
# 
# Totalconversions is the number of users who have clicked the ad and have made a purchase or installed an app
# approved_conversions tells how many became actual active users | Numerica

# ### Observation: We have columns like "political", 'not_political', 'title' and 'message' of the advertisement; 'created at'; 'lang' for languages; 'political_probabilty', and 'paid_for_by'.

# In[ ]:


fb.describe()


# ### Descriptive Statistics: Average Dollars Spent on Facebook Ads
# From another dataset.

# In[ ]:


clicks_df = pd.read_csv('/Users/mehrunisaqayyum/Downloads/datasets_104115_247225_data.csv')
clicks_df


# In[ ]:


clicks_df.columns


# In[ ]:


#Used 'clicks' and 'spent' to calculate the cost per click and per ad.

click_count_per_add = clicks_df.clicks.sum()/clicks_df.clicks.count()
click_ratio = (clicks_df.spent.sum()/63)/clicks_df.clicks.count()
ad_spend_count_ratio = (clicks_df.spent.sum()/63)/clicks_df.clicks.count()
print('Total number of Ads purcheased by IRA from 2015 to 2017: ', clicks_df.clicks.count())
print('Total dollar spent by IRA from 2015 to 2017: ${:.2f}'.format(clicks_df.spent.sum()/63))
print('Average cost per Ad: ${:.2f}'.format(ad_spend_count_ratio))
print('Average number of clicks per Ad: {:.2f}'.format(click_count_per_add))
print('Avg cost per click: ${:.2f}'.format(click_ratio))


# #### Observation
# We see that we have 162,314 records of political ad data on Facebook. The last ten sample how ads were purchased by nonpartisan groups, like "League of Conservation Voters", political organizer groups, like "Indivisible Project", unions "AFT", and international nonprofits like "International Rescue Committee". 
# 

# ### Parse the Text

# ### Pickle Data on Facebook ads

# In[ ]:


# Let's pickle it for later use
data_df.to_pickle("corpus.pkl")


# ### Identify Stop Words

# In[ ]:


stop=set(stopwords.words('english'))

def build_list(fb,col="title"):
    corpus=[]
    lem=WordNetLemmatizer()
    stop=set(stopwords.words('english'))
    new= fb[col].dropna().str.split()
    new=new.values.tolist()
    corpus=[lem.lemmatize(word.lower()) for i in new for word in i if(word) not in stop]
    
    return corpus


# ## Semantic Analysis: Turning FB message into Vectors
# We note the stop words and review counts of words from tf-idf.

# In[ ]:


# Turning FB message into Vectors
from sklearn.feature_extraction.text import TfidfVectorizer

X_train, X_test = train_test_split(emma_paras, test_size=0.4, random_state=0)

vectorizer = TfidfVectorizer(max_df=0.5, # drop words that occur in more than half the paragraphs
                             min_df=2, # only use words that appear at least twice
                             stop_words='english', 
                             lowercase=False, #don't convert everything to lower case (since proper names are people who are targeted in disinfo campaigns)
                             use_idf=True,#we definitely want to use inverse document frequencies in our weighting
                             norm=u'l2', #Applies a correction factor so that longer paragraphs and shorter paragraphs get treated equally
                             smooth_idf=True #Adds 1 to all document frequencies, as if an extra document existed that used every word once.  Prevents divide-by-zero errors
                            )


#Applying the vectorizer
fb_message_tfidf =vectorizer.fit_transform(text_fb)
print("Number of features: %d" % fb_message_tfidf.get_shape()[1])

#splitting into training and test sets
X_train_tfidf, X_test_tfidf= train_test_split(fb_message_tfidf, test_size=0.4, random_state=0)


#Reshapes the vectorizer output into something people can read
X_train_tfidf_csr = X_train_tfidf.tocsr()

#number of paragraphs
n = X_train_tfidf_csr.shape[0]
#A list of dictionaries, one per paragraph
tfidf_bypara = [{} for _ in range(0,n)]
#List of features
terms = vectorizer.get_feature_names()

print(terms)


# In[ ]:


#for each paragraph, lists the feature words and their tf-idf scores
for i, j in zip(*X_train_tfidf_csr.nonzero()):
    tfidf_bypara[i][terms[j]] = X_train_tfidf_csr[i, j]

#Keep in mind that the log base 2 of 1 is 0, so a tf-idf score of 0 indicates that the word was present once in that sentence.
print('Original sentence:', X_train[5])
print('Tf_idf vector:', tfidf_bypara[5])


# In[ ]:


corpus=build_list(text_fb)
counter=Counter(corpus)
most=counter.most_common()
x=[]
y=[]
for word,count in most[:20]:
    if (word not in stop) :
        x.append(word)
        y.append(count)


# In[ ]:


plt.figure(figsize=(9,7))
sns.barplot(x=y,y=x)
plt.title("most common word in title")


# #### Observations:
# Top 5 most common words are ranked:
#     1) 'Committee'
#     2)'International'
#     3) 'Action' and 'Planned'
#     4) 'Parenthood'
#     
# USA and 'America' are interchangeable--so maybe these are double-counting. 
# 
# 'Democratic' not used as much as nonpolitical word of 'rescue'. However, we do not see 'Republican' or 'GOP'. 
# 
# 'Beto' and'O'Rourke' are occuring at same rate. This is the only person to show up in this political ad data set. 

# ### Most Common 'Paid For By'

# In[ ]:


corpus=build_list(text_fb,"paid_for_by")
counter=Counter(corpus)
most=counter.most_common()
x=[]
y=[]
for word,count in most[:10]:
    if (word not in stop) :
        x.append(word)
        y.append(count)
        
plt.figure(figsize=(9,7))
sns.barplot(x=y,y=x)
plt.title("Most Common Word in 'paid_for_by'")


# ### Figure of 'Number of Top Ad Titles'
# To highlight the ads that we would should review for fake versus true classifier to strenghten business case for impact.

# In[ ]:


def plot_count(feature, title,fb, size=1, show_percents=False):
    f, ax = plt.subplots(1,1, figsize=(4*size, 4))
    total = float(len(fb))
    g = sns.countplot(fb[feature],order = fb[feature].value_counts().index[0:20], palette='Set3')
    g.set_title("Number of {}".format(title))
    if (size > 2):
        plt.xticks(rotation=90, size=10)
    if(show_percents):
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2.,
                   height + 3, '{:1.2f}%'.format(100*height/total),
                   ha="center")
    ax.set_xticklabels(ax.get_xticklabels());
    plt.show()


# In[ ]:


plot_count('title','Top Ad Titles', text_fb, 3.5)
#plt.title("Number of Top Ad Titles")


# ### Figure: Number of Most Popular Messages

# In[ ]:


#Note the number (counts) of message.
plot_count('message','message countplot', text_fb, 3.5)


# ## Bag of Words
# Defining functions to identify most common words and the create features from those words in the text. We need to create a df of the "corpus".

# In[ ]:


#We need to use fb_text_df.
 ##Do we need to create features from Bag of Words and vectorize them for the Random Forest Classifier?
def bag_of_words(text):
    allwords = [token.lemma_
               for token in text
               if not token.is_punct
               and not token.is_stop]
    return [item[0] for item in Counter(allwords).most_common(2000)]
print()


# In[ ]:


def bow_features(sentences, common_words):
    fb_text_df = pd.DataFrame(columns=common_words)
    fb_text_df['text_sentence'] = sentences[0]
    fb_text_df['text_source'] = sentences[1]
    fb_text_df.loc[:,common_words] = 0
    
    for i, sentence in enumerate(df['text_sentence']):
        words = [token.lemma_ 
                for token in sentence
                if (
                    not token.is_punct
                    and not token.is_stop
                    and token.lemma_ in common_words
                )]
        for word in words:
            fb_text_df.loc[i, word] += 1
        if i%100 == 0:
            print('Processing row {}'.format(i))
    return fb_text_d


# In[ ]:


#data means fb_text_df
top_words_in_ads_dict = {}
for c in fb_text_df.columns:
    top = fb_text_df[c].sort_values(ascending=False).head(30)
    top_words_in_ads_dict[c]= list(zip(top.index, top.values))

top_words_in_ads_dict


# ### Frequency of Words
# 

# In[ ]:


fb['message'].dtype


# In[ ]:


fb.message.iloc[1:10]


# In[ ]:


temp_variable = "".join(fb.message.iloc[1:1000])


# ### Convert sentences into numeric vectors.
# We need to transform sentences into NUMERIC vectors so that the vectors can be included in a Random Forest Classifier model, which cannot use string values.
# 
# First we must create list.

# In[ ]:


#List 

message_propoganda_list = []
for topic in [#Need to include topic_raw like physics_raw]:
    for sentence in topic['title','paid_for_by']['docs']: #What is docs?
        message_propoganda_list = message_propoganda_list + sentence['message']  #Used message_propaganda for abstract
        
  


# In[ ]:


#Use spacy to count frequency of words

get_ipython().system('python -m spacy download en_core_web_sm')
nlp = spacy.load('en_core_web_sm')

# All the processing work is done here, so it may take a while.
FB_political_doc = nlp(temp_variable)
#FB_popuarity_doc = nlp()


# In[ ]:


# Cleaned Text before Identifying 'Stopwords'
from collections import Counter

# Utility function to calculate how frequently words appear in the data sets.
def word_frequencies(text, include_stop=True):
    
    # Build a list of words.
    # Strip out punctuation and, optionally, stop words.
    words = []
    for token in text:
        if not token.is_punct and (not token.is_stop or include_stop):
            words.append(token.text)
            
    # Build and return a Counter object containing word counts.
    return Counter(words)
    
# The most frequent words:
political_freq = word_frequencies(temp_variable).most_common(10)
#like_freq = word_frequencies().most_common(10)
print('Political:', political_freq)
#print('Popular:', like_freq)


# ## Sentiment Analysis
# Source: https://github.com/adashofdata/nlp-in-python-tutorial/blob/master/3-Sentiment-Analysis.ipynb Regarding text data, there are a few popular techniques that we'll be going through in the next few notebooks, starting with sentiment analysis. A few key points to note with sentiment analysis.
# 
# TextBlob Module: Linguistic researchers have labeled the sentiment of words based on their domain expertise. Sentiment of words can vary based on where it is in a sentence. The TextBlob module allows us to take advantage of these labels. Sentiment Labels: Each word in a corpus is labeled in terms of polarity and subjectivity (there are more labels as well, but we're going to ignore them for now). A corpus' sentiment is the average of these. Polarity: How positive or negative a word is. -1 is very negative. +1 is very positive. Subjectivity: How subjective, or opinionated a word is. 0 is fact. +1 is very much an opinion. For more info on how TextBlob coded up its sentiment function.

# In[ ]:


fb_df2.loc[0:5, 'message']


# In[ ]:


TextBlob('The Mueller investigation is over').sentiment


# In[ ]:


# Create quick lambda functions to find the polarity and subjectivity of each message
# Terminal / Anaconda Navigator: conda install -c conda-forge textblob

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

fb_df2['polarity'] = fb_df2['message'].apply(pol)
fb_df2['subjectivity'] = fb_df2['title'].apply(sub)
fb_df2


# In[ ]:


#Loop each 'message' value

for token in text

