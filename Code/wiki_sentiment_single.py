# -*- coding: utf-8 -*-

from sentiment_models import *
from sentiment_helpers import get_wikipedia_paragraphs_with_api,split_by_closest_period,id_to_sentiment,Method1,Method2,Method3,Method4,model_name
import numpy as np
import pandas as pd
import os

def analyze_url(url,models,output_filename='analysis.csv',output_path=None,show_progress=True):
    char_limit=2000 #character limit, used to avoid running up against model token limits
    min_para_len=50 #minimum string length for paragraphs. Used to filter out headings
        
    if type(models)==str:
        models=[models]
    
    model_objs=[]
        
    module=__import__('sentiment_models')
    for k in range(len(models)):
        class_ = getattr(module, models[k])
        model_objs.append(class_())
    
    paragraphs=get_wikipedia_paragraphs_with_api(url)
    paragraphs_filtered = [s for s in paragraphs if len(s) >= min_para_len]
    para_lengths = [len(s) for s in paragraphs_filtered]
    
    sentiments_dict={}
    argmax_dict={}
    probs_dict={}
    
    for k in range(len(models)):
        sentiments_dict[k]=[]
        argmax_dict[k]=np.zeros((len(paragraphs_filtered),3))
    
    i=0
    
    while i < len(paragraphs_filtered):
        
        #Split paragraph in half if it exceeds the character length limit
        if len(paragraphs_filtered[i])>char_limit:
            while len(paragraphs_filtered[i])>char_limit:
                par_half1,par_half2=split_by_closest_period(paragraphs_filtered[i])
                paragraphs_filtered=paragraphs_filtered[0:i]+[par_half1,par_half2]+paragraphs_filtered[i+1:len(paragraphs_filtered)]
                para_lengths=para_lengths[0:i]+[len(par_half1),len(par_half2)]+para_lengths[i+1:len(para_lengths)]
                
                for k in range(len(models)):
                    argmax_dict[k]=np.append(argmax_dict[k],np.zeros((1,3)),axis=0)
        
        for k in range(len(models)):
            probs=model_objs[k].predict(paragraphs_filtered[i])              
            predicted_class_id=id_to_sentiment(np.argmax(probs))
            sentiments_dict[k].append(predicted_class_id)
            argmax_dict[k][i,np.argmax(probs)]=1
            
            if i==0:
                probs_dict[k]=probs.reshape(1,-1)
            else:
                probs_dict[k]=np.append(probs_dict[k],probs.reshape(1,-1),axis=0)
        
        if show_progress==True:
            print("{}/{} paragraphs analyzed".format(i+1,len(paragraphs_filtered)))
            
        i+=1
        
    m1_dict={}
    m2_dict={}
    m3_dict={}
    m4_dict={}
    
    header_lst=['Name','URL']
    preds_lst=[url[30:len(url)].replace("_"," "),url]
    
    for k in range(len(models)):
        m1_dict[k]=Method1(sentiments_dict[k])
        m2_dict[k]=Method2(argmax_dict[k],para_lengths)
        m3_dict[k]=Method3(probs_dict[k],para_lengths)
        m4_dict[k]=Method4(argmax_dict[k])
        header_lst=header_lst+['Most common sentiment, {}'.format(model_name(models[k])),
                          'Sentiment with highest weighted vote, {}'.format(model_name(models[k])),
                          'Sentiment with highest weighted probability, {}'.format(model_name(models[k])),
                          'Weighted probability of negative sentiment, {}'.format(model_name(models[k])),
                          'Weighted probability of neutral sentiment, {}'.format(model_name(models[k])),
                          'Weighted probability of positive sentiment, {}'.format(model_name(models[k])),
                          'Number of paragraphs with negative sentiment, {}'.format(model_name(models[k])),
                          'Number of paragraphs with neutral sentiment, {}'.format(model_name(models[k])),
                          'Number of paragraphs with positive sentiment, {}'.format(model_name(models[k])),
                          'Percent of paragraphs with negative sentiment, {}'.format(model_name(models[k])),
                          'Percent of paragraphs with neutral sentiment, {}'.format(model_name(models[k])),
                          'Percent of paragraphs with positive sentiment, {}'.format(model_name(models[k]))]
        preds_lst=preds_lst+[m1_dict[k],m2_dict[k],m3_dict[k][0],
                             m3_dict[k][1][0,0],m3_dict[k][1][0,1],m3_dict[k][1][0,2],
                             m4_dict[k][0][0,0],m4_dict[k][0][0,1],m4_dict[k][0][0,2],
                             100*m4_dict[k][1][0,0],100*m4_dict[k][1][0,1],100*m4_dict[k][1][0,2]]
        
    df_out=pd.DataFrame([preds_lst],columns=header_lst)
    
    if output_path!=None:
        os.chdir(output_path)
    
    df_out.to_csv(output_filename,index=False)