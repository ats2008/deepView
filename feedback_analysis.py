
# coding: utf-8

# # Identifying Relevant Issues from Unstructured Reviews

# Unstructred reviews were fed to <b>Watson Natural Language Understanding</b> for congnitive concept identification.
#      
# The Watson showed significant improvement over the <b>VEDER</b> sentiment analysis tool which was almost a complete   failure in identifying context specific polarity scores.
#     
# The keywords identified through the above API are clustered to<b> Domain specific buckets</b>. This is done by identifiying closeness of each identified keyword to the bucket. <b>spaCy</b> was used with an pre-trained model for calculating distant metrics. [ with more time a model fitting our domain can be trained and the process of matching keword to true bins can be made more robust ]
#     
# The keywords obtained from the analysis were weighed accordingly with a custom scorer function to identify <b>TOP ISSUES</b>.
#     
#    
# 

# #### Using Watson Module for keywords as well as topic mining

# In[1]:

from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1        import Features, EntitiesOptions, KeywordsOptions
    
import requests
import json


# Using the API to gather information

# In[11]:
print("importing requiered packeges")
natural_language_understanding = NaturalLanguageUnderstandingV1(
  username='e6421be7-e246-4306-b256-6d01952c8e96',
  password='00u28wsmiF8F',
  version='2017-02-27')
try:
    file=open("commentdb.csv",'r')
except IOError:
    print(" Analysis Begins ")
else:
    l=file.readline()
    n=0
    while l:
        print(" Analyzing review ",n+1,end="..")
        response = natural_language_understanding.analyze(
        text=l,
        features=Features(
        entities=EntitiesOptions(
          emotion=True,
          sentiment=True,
          ),
        keywords=KeywordsOptions(emotion=True,sentiment=True)))
        fan=open(".analysis/an" +str(n)+".json","w")
        json.dump(response,fan,indent=2)
        fan.close()
        print(response['usage'])
        l=file.readline()
        n+=1

file.close()
count_review=n


# ### File Handling stuff to extract useful fields from generated dictionaries
# Clubbing redundant keywords to one with a normalised sentimentvalue and relevence.Also removing entities from our keywords as it doesn't provide any sentiment value.

# In[15]:

from numpy import argsort

keyword=[]
item_keyword=[]
entities=[]
keyword_relevence=[]
keyword_emotions=[]
keyword_sentiment_label=[]
keyword_sentiment_score=[]
entities=[]

print("SAVING AND SORTING KEYWORDS OUT")
for j in range(count_review):
    try:
        f=open(".analysis/an"+str(j)+".json","r")
    except IOError: 
        print(j)
    else:    
        g=json.load(f)
        for ent in g['entities']:
            entities.append(ent['text'])
        text=[]
        text_relevence=[]
        text_emotions=[]
        text_sentiment_label=[]
        text_sentiment_score=[]
        for i in g["keywords"]:
            text.append(i['text'])
            text_relevence.append(i['relevance'])
            text_sentiment_label.append(i['sentiment']['label'])
            text_sentiment_score.append(i['sentiment']['score'])
            if 'emotion' not in sorted(i):
                    text_emotions.append(dict())
            else:
                text_emotions.append(dict(i['emotion']))
        item_keyword.append(text)
        for i in text:
            keyword.append(i)  
        for i in text_relevence:
            keyword_relevence.append(i)  
        for i in text_emotions:
            keyword_emotions.append(i)  
        for i in text_sentiment_label:
            keyword_sentiment_label.append(i) 
        for i in text_sentiment_score:
            keyword_sentiment_score.append(i)  
        f.close()

ent_kew_intercec=[]
print("len of unique kewords obtained : ",len(keyword))
for i in range(len(keyword_relevence)):
    if keyword[i] in entities:
        ent_kew_intercec.append(i)
l=len(ent_kew_intercec)
#print("len of intersection : ",len(ent_kew_intercec))
for i in range(l):
    t=ent_kew_intercec.pop()
    keyword_relevence.pop(t)
    keyword.pop(t)
    keyword_sentiment_label.pop(t)
    keyword_sentiment_score.pop(t)
#print("len of purned kewo : ",len(keyword))

uniq_kwords=set(keyword)
#print("len unq kewords  = ",len(uniq_kwords))
key_word_dict=dict()
for i in uniq_kwords:
    key_word_dict.update({i:{'relavance':[],'label':[],'score':[]}})
#print("len keyw dict ->",len(key_word_dict))

for i,kw in zip(range(len(keyword)),keyword):
    key_word_dict[kw]['relavance'].append(keyword_relevence[i])
    key_word_dict[kw]['label'].append(keyword_sentiment_label[i])
    key_word_dict[kw]['score'].append(keyword_sentiment_score[i])


# Saving to disk for further Analysis 

# #### Metric For Scoring the Keywords

#     The metric used was emperically [ofcourse with a little intuition ! ] formulated from relevance and sentiment value of a keyword.More elaborate, accutare and robust scoring metric could be formulated with a little more work using ML pardigms itself.. [ eg. hyperparameter fitting for a linear combination of emotion scores, sentiment scores and relevance ] 
#     

# In[16]:
print("CALCULATING THE METRIC FOR SENTIMENT SCORE")
tkwords,tscore=[],[]
kwords,score=[],[]
for kwd in sorted(key_word_dict):
    tot=0
    for i,j in zip(key_word_dict[kwd]['score'],key_word_dict[kwd]['relavance']):
        
        tot+=1*i-2*j
    tkwords.append(kwd)
    tot=tot/(3*len(key_word_dict[kwd]['score']))
    tscore.append(tot)

srt=argsort(tscore)
for i in srt:
    kwords.append(tkwords[i])
    score.append(tscore[i])
scored_key=kwords
scored_value=score


print("Top 10 keywords and their Score")

# In[17]:

for i,j,n in zip(scored_key,scored_value,range(10)):
    print(i," \t:\t",j)


# ### Topic Mining from The Keywords and Emotions/Sentiments Associated with it

# Using <b>spaCy</b> for the same

# In[18]:
print("IMPORTING DICTIONARY AND OTHEL NLP TOOLS ")
import spacy 
nlp=spacy.load('en_core_web_md')


# <b>Retrieving the generic Domain Specific Buckets</b>
# 
#     Basic Domain knowledge goes into our model here , identifiying key categories to look into

# In[16]:

# topic_list={'OVERALL':['RECOMMEND', 'EXPERIANCE','VALUE']}
# topic_list.update({'METRO':['METRO', 'METRO RAIL','TRAIN','RAIL','METRO STATION']})
# topic_list.update({'SERVICES':['CUSTOMER CARE', 'COMPLINTS','SECURITY','SMART CARD','PAYMENT','METRO TICKET']})
# topic_list.update({'BUS':['BUS TICKET', 'CONDUCTOR','BUS','PAYMENT','TICKET']})
# topic_list.update({'OTHERS':['OTHERS']})
# topic_list.update({'OPERATIONAL':['SCHEDULE', 'TIMINGS','RELIABILITY']})
# topic_list.update({'LOGISTICS':['BUS STOPS', 'BUS STATIONS','METRO STATIONS','BUS SEATS','SEATS','TICKETING MACHIENE']})



# In[20]:

# generic_bucket_delhi=dict()
# for i in topic_list.keys():
#     generic_bucket_delhi.update({i:{}})
#     for j in topic_list[i]:
#         generic_bucket_delhi[i].update({j:{'keyword': [], 'score': []}}) 


# In[19]:

generic_bucket_delhi={'BUS': {'BUS': {'keyword': [], 'score': []},
  'BUS TICKET': {'keyword': [], 'score': []},
  'CONDUCTOR': {'keyword': [], 'score': []},
  'PAYMENT': {'keyword': [], 'score': []},
  'TICKET': {'keyword': [], 'score': []}},
 'LOGISTICS': {'BUS SEATS': {'keyword': [], 'score': []},
  'BUS STATIONS': {'keyword': [], 'score': []},
  'BUS STOPS': {'keyword': [], 'score': []},
  'METRO STATIONS': {'keyword': [], 'score': []},
  'SEATS': {'keyword': [], 'score': []},
  'TICKETING MACHIENE': {'keyword': [], 'score': []}},
 'METRO': {'METRO': {'keyword': [], 'score': []},
  'METRO RAIL': {'keyword': [], 'score': []},
  'METRO STATION': {'keyword': [], 'score': []},
  'RAIL': {'keyword': [], 'score': []},
  'TRAIN': {'keyword': [], 'score': []}},
 'OPERATIONAL': {'RELIABILITY': {'keyword': [], 'score': []},
  'SCHEDULE': {'keyword': [], 'score': []},
  'TIMINGS': {'keyword': [], 'score': []}},
 'OTHERS': {'OTHERS': {'keyword': [], 'score': []}},
 'OVERALL': {'EXPERIANCE': {'keyword': [], 'score': []},
  'RECOMMEND': {'keyword': [], 'score': []},
  'VALUE': {'keyword': [], 'score': []}},
 'SERVICES': {'COMPLINTS': {'keyword': [], 'score': []},
  'CUSTOMER CARE': {'keyword': [], 'score': []},
  'METRO TICKET': {'keyword': [], 'score': []},
  'PAYMENT': {'keyword': [], 'score': []},
  'SECURITY': {'keyword': [], 'score': []},
  'SMART CARD': {'keyword': [], 'score': []}}}


# In[20]:

buckets=generic_bucket_delhi


# In[21]:

bkt_tops=sorted(buckets)


# #### This function identifies the buckets and sub-buckets to which each keyword will be mapped

# In[22]:
print("SORTING OUT THE KEWORDS INTO CLASSES")
def get_desitination(word,buks):
    shrt_dist=-1000
    gp=""
    sub_group=""
    tkn=nlp(word)
    for topic in bkt_tops:
        dist=-1000
        sub_gp=""
        for subs in sorted(buks[topic]):
            if 'OTHERS' in subs:
                continue
            tkn_subs=nlp(subs)
            tdist=tkn.similarity(tkn_subs)
            
            if dist<tdist:
                dist=tdist
                sub_gp=subs
            #print(word,"<->",subs,"->",tdist)
        if shrt_dist<dist:
            shrt_dist=dist
            sub_group=sub_gp
            gp=topic
    if shrt_dist<0.2:
        gp,sub_group="OTHERS","OTHERS"
    return gp,sub_group
            


# Creating various data structures for further use 

# In[23]:

keywords=scored_key
scores=scored_value

print("flushing the old buckets ...",end="")
for topic in sorted(buckets):
    for sub_topic in sorted(buckets[topic]):
        buckets[topic][sub_topic]['keyword'].clear()
        buckets[topic][sub_topic]['score'].clear()
print("...Done")
l=len(keywords)
t=0
for kwd,scr in zip(keywords,scores):
    t+=1
    a,b=get_desitination(kwd,buckets)
    if t%5==0:
        print(t," / ",l," ",kwd,"->",[a,b])
    buckets[a][b]['keyword'].append(kwd)    
    buckets[a][b]['score'].append(scr)
print("creating keyword Bucket ...",end="")
keyword_bucket=dict()
for topic in sorted(buckets):
    for sub_topic in sorted(buckets[topic]):
        for i in buckets[topic][sub_topic]['keyword']:
            keyword_bucket.update({i:[topic,sub_topic]})
print("..DONE")


# Saving the Bucket statistics

# In[29]:

file=open("RESULTS/TopicMined.csv",'w')
for topic in sorted(buckets):
    file.write("\n\n"+topic+"\n")
    for sub_topic in sorted(buckets[topic]):
        s=sum(buckets[topic][sub_topic]['score'])
        file.write(sub_topic+","+str(s)+"\t"+str(len(buckets[topic][sub_topic]['score']))+"\n")
       # print(sub_topic+str(s)+"\n")
file.close()


print("PLOTING STATISTICS FOR INFERENCES")

# In[28]:

import matplotlib.pyplot as plt
import plotly.plotly as py

sentscore=[]
senttopic=[]
sentfreq=[]
sentsubtopic=[]
sentsubscore=[]
sentsubfreq=[]    
for topic in sorted(buckets):
    if topic !='OTHERS':
        a,b,d=0,0,0
        sentsubtopic=[]
        sentsubscore=[]
        sentsubfreq=[]    
        for sub_topic in sorted(buckets[topic]):
                s=sum(buckets[topic][sub_topic]['score'])
                b=len(buckets[topic][sub_topic]['score'])
                if(b!=0):
                    a=a+(s/b)
                    sentsubscore.append(-s/b)
                    d=d+len(buckets[topic][sub_topic])
                    sentsubtopic.append(sub_topic)
                    sentsubfreq.append(b)
        f=plt.figure(figsize=(10, 5))    
        plt.bar(list(range(len(sentsubtopic))),sentsubfreq, width=.5, align="center")
        plt.title(topic+' Frequecy')
        plt.ylabel('Frequency')
        plt.xticks(list(range(len(sentsubtopic))),sentsubtopic)
#        plt.show()
        f.savefig("RESULTS/"+topic+" Frequency.png")        
        plt.figure(figsize=(10,5))  
        plt.bar(list(range(len(sentsubtopic))),sentsubscore, width=.5, align="center")
        plt.title(topic+' Sentiment Score')
        plt.ylabel('Sentiment Score')
        plt.xticks(list(range(len(sentsubtopic))),sentsubtopic)
 #       plt.show()
        c=-a/len(buckets[topic])
        senttopic.append(topic)
        sentscore.append(c)
        sentfreq.append(d)
        f.savefig("RESULTS/"+topic+" Sentiment Score.png")

plt.figure(figsize=(5,5 ))  
plt.bar(list(range(6)),sentscore, width=.5, align="center")
plt.xticks(list(range(6)),senttopic)
plt.title('Overall Sentiment Score')
#plt.show()
f.savefig("RESULTS/Overall Sentiment Score.png")
print("look into the RESULT DIRECTORY for NEW RESULTS")






