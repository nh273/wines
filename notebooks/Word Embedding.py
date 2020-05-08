
# coding: utf-8

# In[1]:


import pandas as pd
import gensim


# In[2]:


model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True) 
model.vector_size

