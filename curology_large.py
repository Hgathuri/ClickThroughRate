
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
from sklearn.linear_model import LinearRegression
import json, re, ast, difflib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer
get_ipython().run_line_magic('matplotlib', 'inline')

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.linear_model import LinearRegression

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor


# In[2]:


df = pd.read_csv('FB%2FIG.csv',encoding = 'Latin')


# In[ ]:


list(df.columns)


# In[3]:


len(df.columns)


# ## Feature Extraction
# 
# We pull data from json objects.

# In[4]:


NLPvars = ['NLP - Link Description','NLP - Title','NLP - Body','NLP - Video Transcription']
NLPkeys = ['sentences', 'tokens', 'entities', 'documentSentiment', 'language', 'categories']

VIDvars = ['Video Intelligence Data']
Other = ['Targeting']

def format_json(string,var,index):
    if type(string) == float:
        if np.isnan(string) == True:
            string = 'SKIP'
    else:
        try:
            if (type(eval(string)) == list) | (type(string) == list):
                string = 'LIST SKIP'
            elif type(eval(string)) == dict:
                d = json.loads(string)

        except:
            try:
                d = json.loads(string)
            except:
                try:
                    string = string.replace(';',',').replace("\\n", "").replace("\t", "").replace(" ", "").replace('\\', '')
                    d = json.loads(string)
                except:
                    try:
                        string = string[0]+string[-1]+string[1:-1]
                        d = json.loads(string)
                    except:
                        print('JSON parsing error @ '+var+' at index '+str(index))
                        string = 'SKIP'
    return(string)




def NLPextract(df):
    
    for var in NLPvars:
        for index, row in df.iterrows():
            print(index)
            string = format_json(row[var],var,index)
            if string !='SKIP':
                
                d = json.loads(string)
                df.loc[index,var+'_sentiment_mag_mean'] = np.mean([x['sentiment']['magnitude'] for x in d['sentences'] if 'sentiment' in x])
                df.loc[index,var+'_sentiment_score_mean'] = np.mean([x['sentiment']['score'] for x in d['sentences'] if 'sentiment' in x])
                df.loc[index,var+'_documentSentiment_mag'] = d['documentSentiment']['magnitude']
                
                df.loc[index,var+'_documentSentiment_score'] = d['documentSentiment']['score']
                
                df.loc[index,var+'_documentSentiment_language'] = d['language']
                
                df.loc[index,var+'_entities'] = str([x['name'] for x in d['entities']])
                
    return(df)

def VIDextract(df):
    for index, row in df.iterrows():
        
        df.loc[index,'VID_shot_entities'] = np.nan
        df.loc[index,'VID_shot_catentities'] = np.nan
        df.loc[index,'VID_segment_entities'] = np.nan
        df.loc[index,'VID_segment_catentities'] = np.nan
        df.loc[index,'VID_all_entities'] = np.nan
        df.loc[index,'VID_all_catentities'] = np.nan
        
        string = format_json(row['Video Intelligence Data'],'Video Intelligence Data',index)
        if string == 'LIST SKIP':
            string = eval(row['Video Intelligence Data'])
            
            VID_all_entities = str(set([x['entity']['description'] for x in string]))
            df.loc[index,'VID_all_entities'] = VID_all_entities
            
            cat = [x['categoryEntities'] for x in string if 'categoryEntities' in str(x)]
            if len(cat) == 1:
                VID_all_catentities = str(set([x['description'] for x in cat[0]]))
                df.loc[index,'VID_all_catentities'] = VID_all_catentities
            elif len(cat) > 1:
                VID_all_catentities = str(set([x[0]['description'] for x in cat]))
                df.loc[index,'VID_all_catentities'] = VID_all_catentities
            #except:
            #    pass
        elif string != 'SKIP':

            d = json.loads(string)
            try:
                VID_segment_entities = [x['entity']['description'] for x in d['segmentLabelAnnotations']]
                df.loc[index,'VID_segment_entities'] = str(VID_segment_entities)
            except:
                pass
            try:
                VID_segment_time = [x['segments'][0]['segment']['endTimeOffset'] for x in d['segmentLabelAnnotations']][0]
                df.loc[index,'VID_segment_time'] = str(VID_segment_time)
            except:
                pass
            try:    
                VID_segment_catentities = [x['categoryEntities'][0]['description'] for x in d['segmentLabelAnnotations'] if 'categoryEntities' in x]
                df.loc[index,'VID_segment_catentities'] = str(VID_segment_catentities)
            except:
                pass
            try:
                VID_shot_entities = [x['entity']['description'] for x in d['shotLabelAnnotations']]
                df.loc[index,'VID_shot_entities'] = str(VID_shot_entities)
            except:
                pass
            try:
                VID_shot_catentities = [x['categoryEntities'][0]['description'] for x in d['shotLabelAnnotations'] if 'categoryEntities' in x]
                df.loc[index,'VID_shot_catentities'] = str(VID_shot_catentities)
            except:
                pass
            


            try:
                
                VID_all_entities = set(VID_shot_entities + VID_segment_entities)
                
                df.loc[index,'VID_all_entities'] = str(VID_all_entities)
            except:
                try:
                    VID_all_entities = set(VID_shot_entities)
                    df.loc[index,'VID_all_entities'] = str(VID_all_entities)
                except:
                    VID_all_entities = set(VID_segment_entities)
                    df.loc[index,'VID_all_entities'] = str(VID_all_entities)                    
            
            
            try:
                VID_all_catentities = set(VID_shot_catentities + VID_segment_catentities)
                df.loc[index,'VID_all_catentities'] = str(VID_all_catentities)
            except:
                try:
                    VID_all_catentities = set(VID_shot_catentities)
                    df.loc[index,'VID_all_catentities'] = str(VID_all_catentities)
                except:
                    VID_all_catentities = set(VID_segment_catentities)
                    df.loc[index,'VID_all_catentities'] = str(VID_all_catentities)

    return(df)
    
def Targetingextract(df):
    for index, row in df.iterrows():
        if df['Targeting'][index] != 'No Data':
            if type(df['Targeting'][index]) == str:
                d = json.loads(df['Targeting'][index].replace(';',','))
                if type(d) == list:
                    d = d[0]
                for i in list(d.keys()):
                    try:
                        df.loc[index,'Targeting_'+i] = str(d[i])
                    except:
                        df.loc[index,'Targeting_'+i] = str(d[i])
    return(df)


# In[ ]:


list(df.columns)


# In[5]:


df = NLPextract(df)
df = VIDextract(df)
df = Targetingextract(df)


# In[6]:


len(list(df))


# In[7]:


df['Account Name'].notnull().sum()


# In[8]:


df.head()


# In[ ]:


df.tail()


# In[9]:


ax = df.notnull().sum().plot(kind='bar',figsize=(20,20))
ax.set_xlabel("variable")
ax.set_ylabel("count")


# We'll remove features with few non-null values. 

# In[10]:


varlist = list(df)

for i in list(df):
    if df[i].notnull().sum() < 15:
        varlist.remove(i)


# In[11]:


varlist = list(df)

for i in list(df):
    if df[i].notnull().sum() < 15:
        a = varlist.remove(i)
        print(a)


# The distributions of 'CPP' and 'CTR' are right-skewed.

# In[12]:


df['CPP'] = pd.to_numeric(df['CPP'], errors='coerce')
df['CTR'] = pd.to_numeric(df['CTR'], errors='coerce')


# In[13]:


len(list(df))


# In[14]:


len(varlist)


# In[15]:


len(df[i])


# In[16]:


df['CPP'].hist()


# In[17]:


df['CTR'].hist()


# We remove 0 variance features.

# In[18]:


removelist = []
for i in list(df):
    if len(df[i].value_counts()) == 1:
        print(i)
        removelist.append(i)
df = df[[x for x in list(df) if x not in removelist]]


# In[19]:


print(removelist)


# In[20]:


df['Type'].value_counts()


# In[22]:


len(removelist)


# In[23]:


len(df.columns)


# In[24]:


for i in list(df):
    print(list(df[i].value_counts()))


# In[25]:


df.columns


# In[26]:


df['Ad ID'].value_counts()


# In[27]:


def listtocol(var):
    listy = []

    for i in df[var]:
        if type(i) == str:
            for i in i.split("'"):
                if i not in [', ',']','[']:
                    listy.append(i)

    listy = [re.sub(r'[^ a-zA-Z0-9.\-]', "", x) for x in listy]
            
    for i in set(listy):
        newvar = var+'_'+i
        df[newvar] = 0
        df.loc[df[var].str.contains(i)==True,newvar] = 1
    
    return(df)


# In[28]:


listvars = ['NLP - Link Description_entities','NLPlink_mood','NLP - Body_entities','NLPbody_mood','NLP - Video Transcription_entities',
           'VID_all_entities','VID_all_catentities']

listvars = [x for x in listvars if x in list(df)]


# We create binary variables from mood and entity list features. 

# In[29]:


for i in listvars:
    print(i)
    listtocol(i)


# In[30]:


listvars


# In[31]:


df.VID_all_catentities


# We'll gather the binary variables.

# In[32]:


dummies = []

for i in listvars:
    for j in list(df):
        if i+'_' in j:
            dummies.append(j)


# In[33]:


list(df)


# In[34]:


dummies


# We remove binary features with only singular positive outcomes. 

# In[35]:


remove2 = []

for i in dummies:
    if len(df[i].value_counts()) == 1:
        print(i)
        remove2.append(i)
    if df[i].sum() == 1:
        remove2.append(i)
        
dummies = [x for x in dummies if x not in remove2]
df = df[[x for x in list(df) if x not in remove2]]


# In[36]:


df['VID_segment_time']


# In[37]:


df['VID_segment_time'] = df['VID_segment_time'].str.replace('s', '').astype(float)


# In[40]:


df['gender'] = np.nan
df.loc[df['Targeting_genders'] == '[1]','gender'] =1
df.loc[df['Targeting_genders'] == '[2]','gender'] =0

df['ages'] = 'age_group_'+df['Targeting_age_min'] +'-'+ df['Targeting_age_max']
df = pd.concat([df, pd.get_dummies(df['ages'])], axis=1)

#xvars = ['NLPlink_sentiment_mag_mean','NLPlink_sentiment_score_mean','NLPlink_documentSentiment_mag',
#        'NLPlink_documentSentiment_score','NLPbody_sentiment_mag_mean','NLPbody_sentiment_score_mean',
#        'NLPbody_documentSentiment_mag','NLPbody_documentSentiment_score','NLPtext_sentiment_mag_mean',
#        'NLPtext_sentiment_score_mean','NLPtext_documentSentiment_mag','NLPtext_documentSentiment_score','gender',
#        'VID_segment_time']

xvars = ['NLP - Link Description_sentiment_mag_mean',
 'NLP - Link Description_sentiment_score_mean',
 'NLP - Link Description_documentSentiment_mag',
 'NLP - Link Description_documentSentiment_score',
 'NLP - Title_sentiment_mag_mean',
 'NLP - Title_sentiment_score_mean',
 'NLP - Title_documentSentiment_mag',
 'NLP - Title_documentSentiment_score',
 'NLP - Body_sentiment_mag_mean',
 'NLP - Body_sentiment_score_mean',
 'NLP - Body_documentSentiment_mag',
 'NLP - Body_documentSentiment_score',
 'NLP - Video Transcription_sentiment_mag_mean',
 'NLP - Video Transcription_sentiment_score_mean',
 'NLP - Video Transcription_documentSentiment_mag',
 'NLP - Video Transcription_documentSentiment_score','gender','VID_segment_time']

dummies = dummies + [x for x in list(df) if 'age_group_' in x]


# Many of the dummy variables are very sparse. In this sample, we will only consider denser variables but in larger iterations we will use topic modeling to create dense thematic variables. 

# In[41]:


for i in dummies:
    if df[i].value_counts(normalize=True)[0] >.90:
        dummies = [x for x in dummies if x != i]


# In[42]:


for i in dummies:
    if len(df.loc[df[i].notnull()])/len(df) <.99:
        dummies = [x for x in dummies if x!=i]
        
for i in xvars:
    if len(df.loc[df[i].notnull()])/len(df) <.75:
        xvars = [x for x in xvars if x!=i]


# ## Model Generation/Feature Selection via Sequential Features Selection
# 
# We will evaluate the performance of using k-sized subsets of features when predicing the 'CPP' and 'CTR' outcomes.

# In[43]:


def rmse(y, y_pred):
    return(np.sqrt(np.mean((y_pred - y)**2)))
rmse_scorer = make_scorer(rmse, greater_is_better=False)


# In[44]:


xvars


# In[45]:


dftemp = df[['CPP','CTR']+dummies+xvars]

xvars2 = [x for x in xvars if len(dftemp[x].value_counts())>1]
dummies2 = [x for x in dummies if len(dftemp[x].value_counts())>1]

dftemp = df[['CPP','CTR']+dummies2+xvars2]
dftemp.dropna(inplace=True, axis=0)

Xvars = [x for x in list(dftemp) if x not in  ['CPP','CTR']]


# In[46]:


len(dftemp)


# In[47]:


dftemp = dftemp.loc[dftemp['CTR']!=0]
len(dftemp)


# In[48]:


dftemp = dftemp.loc[dftemp['CPP'].notnull()]
print(len(dftemp))
print(len(list(dftemp)))


# In[49]:


sfs = SFS(LinearRegression(), 
          k_features=30, 
          forward=True, 
          floating=True, 
          scoring=rmse_scorer,
          cv=4)

sfs = sfs.fit(dftemp[Xvars].values, dftemp['CPP'].values)

sfs_LR = sfs.get_metric_dict()

print('LR COMPLETE')


# We run a sequentual feature selector with various models (Linear Regression and Support Vector Regression with varying kernel parameters). The selector determines the best 1 feature model, 2 feature model, etc. and records the Root Mean Squared Error (RMSE) (our performance measure for each). We observe the results of a Linear Regression model and run the same analysis for CPP and CTR.

# In[ ]:


for i in ['sfs_LR']:
    fig1 = plot_sfs(eval(i), kind='std_dev')

    plt.title('Sequential Forward Selection (w. StdDev) - '+i)
    plt.grid()
    plt.show()


# In[ ]:


best = sfs_LR[10]['feature_idx']

listy = dummies2+xvars2
testlist = []
for i in best:
    testlist.append(listy[i])
    print(listy[i])


# We observe the coefficients produced by the LR model below, representing the relationship with the output, 'CPP' variable. 

# In[ ]:


[x for x in list(df) if 'age_group_']


# In[ ]:


LR = LinearRegression()
coeffs = LR.fit(dftemp[testlist].values, dftemp['CPP'].values).coef_
print('const:')
print(LR.intercept_)
pd.Series(coeffs, index = testlist).plot('bar',figsize =(8,8))


# For our binary features, we will look at how the mean of 'CPP' varies by the presence of an indicator.

# ## Sample Actionable Insights - CPP
# 
# **Based on the results of this analysis, we can lower cost per purchase by:**
# 
# **-Including the following words in the Link text:**
# * life
# 
# **-Including the following words in the video transcription:**
# * Sephora
# * nothing
# 
# **-Excluding the following entities in the video transcription:**
# * solution
# * cure
# * outside
# * neck
# 
# **-Including the following entities in the video:**
# * text messaging
# 
# **-Excluding the following entities in the video:**
# * facial expression
# 
# **-Targeting the following age group:**
# * 13-34

# In[ ]:


dftemp = dftemp.loc[dftemp['CTR'].notnull()]
print(len(dftemp))
print(len(list(dftemp)))


# In[ ]:


sfs = SFS(LinearRegression(), 
          k_features=30, 
          forward=True, 
          floating=True, 
          scoring=rmse_scorer,
          cv=4)

sfs = sfs.fit(dftemp[Xvars].values, dftemp['CTR'].values)

sfs_LR = sfs.get_metric_dict()

print('LR COMPLETE')


# In[ ]:


for i in ['sfs_LR']:
    fig1 = plot_sfs(eval(i), kind='std_dev')

    plt.title('Sequential Forward Selection (w. StdDev) - '+i)
    plt.grid()
    plt.show()


# In[ ]:


best = sfs_LR[7]['feature_idx']

listy = dummies2+xvars2
testlist = []
for i in best:
    testlist.append(listy[i])
    print(listy[i])
testlist = testlist[:-2]


# In[ ]:


LR = LinearRegression()
coeffs = LR.fit(dftemp[testlist].values, dftemp['CTR'].values).coef_
print('const:')
print(LR.intercept_)
pd.Series(coeffs, index = testlist).plot('bar',figsize =(8,8))


# ## Sample Actionable Insights - CTR
# 
# **Based on the results of this analysis, we can increase click through rate by:**
# 
# **-Including the following words in the Link text:**
# * trial
# 
# **-Including the following words in the video transcription:**
# * nothing
# 
# **-Including the following entities in the video:**
# * product
# * facial expression
# 
# **-Excluding the following entities in the video:**
# * cream
