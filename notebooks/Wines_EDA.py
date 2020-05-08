
# coding: utf-8

# In[1]:


import nltk
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import bokeh
import matplotlib.pyplot as plt
import pylab as pl
from ast import literal_eval
import string
import random
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.stem import WordNetLemmatizer


# In[2]:


DATA_MAG = "../data/raw/winemag-data-130k-v2.csv"


# In[3]:


df = pd.read_csv(DATA_MAG, index_col=0)
df.shape


# In[4]:


df = df.drop_duplicates()
df.shape


# In[5]:


# Just a minor terminology correction
df.rename(columns={'variety': 'varietal'}, inplace=True)


# In[6]:


df.hist(column = 'price', bins=100, log=True)


# In[7]:


# Nice bell-curve looking kinda distribution
df.hist(column = 'points', bins=100)


# In[8]:


df['unique_taster'] = df['taster_name'] + df['taster_twitter_handle'].fillna('__notwitter')


# In[9]:


df['unique_taster'].value_counts()


# In[10]:


# a few countries produce most of the wines here
fig, ax = plt.subplots()
df['country'].value_counts().plot(ax=ax, kind='bar')


# In[11]:


df['province'].value_counts()[:10]


# In[12]:


df['varietal'].value_counts()[:50]


# Uh-oh. Looks like there are 2 big confusions: 'Syrah' & 'Shiraz' are the same grape. Something called 'Sauvignon', which should really be 'Sauvignon Blanc'. Let's just check to be sure that it's not 'Sauvignon Vert' or sth.

# In[13]:


df[df['varietal']=='Sauvignon']['province'].value_counts()


# In[14]:


df[df['varietal']=='Sauvignon']['taster_name'].value_counts()


# In[15]:


df[(df['varietal']=='Sauvignon') & (df['taster_name']=="Kerin O’Keefe")][['title','description']]


# Googling some of these bottles showed that they are indeed Sauvignon Blanc.

# In[16]:


df['varietal'] = df['varietal'].map(lambda x: 'Sauvignon Blanc' if x == 'Sauvignon' else x)
df['varietal'] = df['varietal'].map(lambda x: 'Shiraz' if x == 'Syrah' else x)


# The average (mean) province contributes about 300 wines to the list. But obviously some provinces dominate.

# In[17]:


df.province.value_counts().nlargest(25)


# In[18]:


df.groupby(['province', 'varietal'])['description'].count().nlargest(50)


# In[19]:


df['title'].value_counts().hist(bins=20, log=True)


# Most wines appear only once, but seems like a few appears multiple times

# In[20]:


df.groupby(by='country')['designation'].value_counts().nlargest(20)


# The designation doesn't seem to be anything regulated like AOC or AOP or DOCG

# In[21]:


df['country_varietal'] = df['country']+ ' ' +df['varietal']
df['province_varietal'] = df['province'] + ' ' + df['varietal']


# In[22]:


top_grape = df['varietal'].value_counts().nlargest(50).to_frame('count').reset_index()
top_grape.rename(columns={'index':'varietal'}, inplace=True)
top_grape


# In[23]:


top_grape_country = df['country_varietal'].value_counts().nlargest(75).to_frame('count').reset_index()
top_grape_country.rename(columns={'index':'country_varietal'}, inplace=True)


# In[39]:


country_df = df.groupby(by='country').agg({'points':'median','price':'median','description':'; '.join}).reset_index()
varietal_df = df.groupby(by='varietal').agg({'points':'median','price':'median','description':'; '.join}).reset_index()
country_varietal_df = df.groupby(by=['country','varietal']).agg({'points':'median','price':'median','description':'; '.join}).reset_index()
country_varietal_df['country_varietal'] = country_varietal_df['country'] + ' ' + country_varietal_df['varietal']
country_varietal_df.head()


# In[40]:


province_varietal_df = df.groupby(by=['province','varietal']).agg({'points':'median','price':'median','description':'; '.join}).reset_index()
province_varietal_df['province_varietal'] = province_varietal_df['province'] + ' ' + province_varietal_df['varietal']


# In[26]:


# obtain grape name tokens for later removal from description
grape_names = df['varietal'].fillna('').unique().tolist()
grape_tokens = []
for name in grape_names:
    name = name.lower()
    tokens = re.split('[- ]', name)
    grape_tokens += tokens


# In[27]:


wine_remove_words = ['wine','rosé', 'red', 'white', 'drink', 'aroma', 'flavor','vineyard', 
                     'zin','acidity', 'structure', 'note']
other_remove_words = ['\'s', "'"]


# In[28]:


post_lem_remove = set(wine_remove_words + other_remove_words)
pre_lem_remove = set([p for p in string.punctuation] + nltk.corpus.stopwords.words('english') + grape_tokens)
#use set to improve performance

def clean_tokenizer(text):
    """Turn long text into lowercase tokens, remove punctuations & stopwords
    e.g: fro"""
    tokens = nltk.tokenize.word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [lemmatizer.lemmatize(t) for t in tokens if not 
                    ((t in pre_lem_remove) or (lemmatizer.lemmatize(t) in post_lem_remove) or (t.isnumeric()))]
    
    return clean_tokens

tfidf = TfidfVectorizer(lowercase=False, tokenizer=clean_tokenizer, token_pattern=None,
                        stop_words=None, max_df=0.7, min_df=10, ngram_range=(1,3))
# discarding terms that occur > 70% of the time & < 10 times. Using both normal & bigrams
tfidf_cv = TfidfVectorizer(lowercase=False, tokenizer=clean_tokenizer, token_pattern=None,
                        stop_words=None, max_df=0.7, min_df=10, ngram_range=(1,3))
tfidf_pv = TfidfVectorizer(lowercase=False, tokenizer=clean_tokenizer, token_pattern=None,
                        stop_words=None, max_df=0.7, min_df=10, ngram_range=(1,3))


# In[29]:


tfidf_matrix = tfidf.fit_transform(varietal_df['description'])
tfidf_matrix_cv = tfidf_cv.fit_transform(country_varietal_df['description'])
tfidf_matrix_pv = tfidf_pv.fit_transform(province_varietal_df['description'])


# In[30]:


tfidf_matrix.shape, tfidf_matrix_cv.shape, tfidf_matrix_pv.shape


# In[31]:


def top_tfidf_feats(tfidf_row, features, n_feats=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    top_ids = np.argsort(tfidf_row)[::-1][:n_feats]
    # sort features in row (by tfidf score) -> ascending
    # then reverse to get descending, then slice top n
    
    top_features = [(features[i], tfidf_row[i]) for i in top_ids]
    df = pd.DataFrame(top_features) # return as a Dataframe
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(tfidf_matrix, features, row_id, n_feats=25):
    ''' Top tfidf features in specific document (matrix row)
    Gotta call this instead of top_tf_idf_feats because tfidf
    is sparse matrix -> does not support all matrix operations.
    '''
    # first convert a row to full matrix format
    row = np.squeeze(tfidf_matrix[row_id].toarray())
    
    # return a df
    return top_tfidf_feats(row, features, n_feats)


# In[52]:


features = tfidf.get_feature_names()
features_cv = tfidf_cv.get_feature_names()
features_pv = tfidf_pv.get_feature_names()

def look_at(var, name, n_feats=10):
    if var=='v':
        tfidf_object = tfidf_matrix; feature_names = features; col = 'varietal'; grapes_df=varietal_df
    elif var == 'cv':
        tfidf_object = tfidf_matrix_cv; feature_names = features_cv; col = 'country_varietal'; grapes_df=country_varietal_df
    elif var == 'pv':
        tfidf_object = tfidf_matrix_pv; feature_names = features_pv; col = 'province_varietal'; grapes_df=province_varietal_df

    row = grapes_df[grapes_df[col]==name]
    row_id = row.index
    return top_feats_in_doc(tfidf_object, feature_names, row_id, n_feats=n_feats)

def look_at_random(n=10, n_feats=10):
    '''Look at n random wines'''
    row_ids = random.sample(range(1, 100), n)
    for row_id in row_ids:
        print('_'*10)
        print(varietal_df.iloc[row_id]['varietal'])
        print(top_feats_in_doc(tfidf_matrix, features,row_id, n_feats=n_feats))

def look_at_random_cv(n=10, n_feats=10):
    '''Look at n random wines'''
    row_ids = random.sample(range(1, 100), n)
    for row_id in row_ids:
        print('_'*10)
        print(country_varietal_df.iloc[row_id]['country_varietal'])
        print(top_feats_in_doc(tfidf_matrix_2, features_2,row_id, n_feats=n_feats))


# In[33]:


# the next 2 lines are a repeat of a previous cell
top_grape = df['varietal'].value_counts().nlargest(57).to_frame('count').reset_index()
top_grape.rename(columns={'index':'varietal'}, inplace=True)

top_grape_names = top_grape['varietal'].tolist()
top_grape_names = [grape for grape in top_grape_names if grape not in 
                   ["Red Blend","White Blend","Sparkling Blend", "Port", "Rosé"]]
print(len(top_grape_names))
for grape in top_grape_names[:10]:
    print('_________________')
    print(grape)
    look_at('v', grape, n_feats=5)


# In[92]:


top_pv = list(df['province_varietal'].value_counts().nlargest(500).index)


# In[93]:


collector = {}
for pv in top_pv:
    collector[pv] = look_at('pv', pv, 20).to_dict(orient='records')


# In[94]:


for pv in top_pv[:20]:
    print("-"*10)
    print(pv)
    print(look_at('pv', pv, 5))


# In[95]:


import json

with open('../data/intermediary/tfidf.json', 'w') as fp:
    json.dump(collector, fp)


# In[82]:


import scipy.sparse
scipy.sparse.save_npz('../data/intermediary/tfidf_matrix.npz', tfidf_matrix_pv)


# In[80]:


with open('../data/intermediary/features.txt', 'w') as f:
    for item in features_pv:
        f.write(f"{item}\n")


# In[81]:


province_varietal_df.to_csv('../data/intermediary/province_varietal.csv', index=False)


# In[96]:


with open('../data/intermediary/top_pv.txt', 'w') as f:
    for pv in top_pv:
        f.write(f"{pv}\n")

