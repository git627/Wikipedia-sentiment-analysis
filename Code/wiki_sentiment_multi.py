# -*- coding: utf-8 -*-

from sentiment_models import *
from sentiment_helpers import get_wikipedia_paragraphs_with_api,split_by_closest_period,id_to_sentiment,Method1,Method2,Method3,Method4,model_name
import numpy as np
import pandas as pd
import os

def analyze_list(urls,models,output_filename='analysis.csv',output_path=None,show_progress=True):
        char_limit=2000 #character limit, used to avoid running up against model token limits
        min_para_len=50 #minimum string length for paragraphs. Used to filter out headings
        
        if type(models)==str:
            models=[models]
        
        model_objs=[]
            
        module=__import__('sentiment_models')
        for k in range(len(models)):
            class_ = getattr(module, models[k])
            model_objs.append(class_())
                        
        for i in range(len(urls)):
        
            paragraphs=get_wikipedia_paragraphs_with_api(urls[i])
            paragraphs_filtered = [s for s in paragraphs if len(s) >= min_para_len]
            para_lengths = [len(s) for s in paragraphs_filtered]
            
            sentiments_dict={}
            argmax_dict={}
            probs_dict={}
            
            for k in range(len(models)):
                sentiments_dict[k]=[]
                argmax_dict[k]=np.zeros((len(paragraphs_filtered),3))
            
            j=0
            
            while j < len(paragraphs_filtered):
                
                #Split paragraph in half if it exceeds the character length limit
                if len(paragraphs_filtered[j])>char_limit:
                    while len(paragraphs_filtered[j])>char_limit:
                        par_half1,par_half2=split_by_closest_period(paragraphs_filtered[j])
                        paragraphs_filtered=paragraphs_filtered[0:j]+[par_half1,par_half2]+paragraphs_filtered[j+1:len(paragraphs_filtered)]
                        para_lengths=para_lengths[0:j]+[len(par_half1),len(par_half2)]+para_lengths[j+1:len(para_lengths)]
                        
                        for k in range(len(models)):
                            argmax_dict[k]=np.append(argmax_dict[k],np.zeros((1,3)),axis=0)
                
                for k in range(len(models)):
                    probs=model_objs[k].predict(paragraphs_filtered[j])                
                    predicted_class_id=id_to_sentiment(np.argmax(probs))
                    sentiments_dict[k].append(predicted_class_id)
                    argmax_dict[k][j,np.argmax(probs)]=1
                    
                    if j==0:
                        probs_dict[k]=probs.reshape(1,-1)
                    else:
                        probs_dict[k]=np.append(probs_dict[k],probs.reshape(1,-1),axis=0)
                    
                #print("{}/{} paragraphs analyzed".format(j+1,len(paragraphs_filtered)))
                    
                j+=1
                
            m1_dict={}
            m2_dict={}
            m3_dict={}
            m4_dict={}
            
            header_lst=['Article length']
            preds_lst=[sum(para_lengths)]
            
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
                
            df_page=pd.DataFrame([preds_lst],columns=header_lst)
            
            if i==0:
                df_outputs=df_page
            else:
                df_outputs=pd.concat([df_outputs,df_page],axis=0)
            
            if show_progress==True:
                print("{}/{} articles analyzed".format(i+1,len(urls)))
            
        df_outputs=df_outputs.reset_index(drop=True)
        
        page_names=list(map(lambda x: x[30:len(x)].replace("_"," "), urls))
        df_in=pd.DataFrame({'Name':page_names,'URL':urls})
            
        df_final=pd.concat([df_in,df_outputs],axis=1)
        
        if output_path!=None:
            os.chdir(output_path)
        
        df_final.to_csv(output_filename,index=False)
        
def analyze_csv(input_filename,models,output_filename='analysis.csv',input_path=None,output_path=None,
            show_progress=True,url_column='URL'):
        char_limit=2000 #character limit, used to avoid running up against model token limits
        min_para_len=50 #minimum string length for paragraphs. Used to filter out headings
        
        if type(models)==str:
            models=[models]
        
        model_objs=[]
            
        module=__import__('sentiment_models')
        for k in range(len(models)):
            class_ = getattr(module, models[k])
            model_objs.append(class_())
        
        if input_path!=None:
            os.chdir(input_path)
            
        df_in=pd.read_csv(input_filename)
        urls=df_in[url_column]
                
        for i in range(len(urls)):
        
            paragraphs=get_wikipedia_paragraphs_with_api(urls[i])
            paragraphs_filtered = [s for s in paragraphs if len(s) >= min_para_len]
            para_lengths = [len(s) for s in paragraphs_filtered]
            
            sentiments_dict={}
            argmax_dict={}
            probs_dict={}
            
            for k in range(len(models)):
                sentiments_dict[k]=[]
                argmax_dict[k]=np.zeros((len(paragraphs_filtered),3))
            
            j=0
            
            while j < len(paragraphs_filtered):
                
                #Split paragraph in half if it exceeds the character length limit
                if len(paragraphs_filtered[j])>char_limit:
                    par_half1,par_half2=split_by_closest_period(paragraphs_filtered[j])
                    paragraphs_filtered=paragraphs_filtered[0:j]+[par_half1,par_half2]+paragraphs_filtered[j+1:len(paragraphs_filtered)]
                    para_lengths=para_lengths[0:j]+[len(par_half1),len(par_half2)]+para_lengths[j+1:len(para_lengths)]
                    
                    for k in range(len(models)):
                        argmax_dict[k]=np.append(argmax_dict[k],np.zeros((1,3)),axis=0)
                
                for k in range(len(models)):
                    probs=model_objs[k].predict(paragraphs_filtered[j])                
                    predicted_class_id=id_to_sentiment(np.argmax(probs))
                    sentiments_dict[k].append(predicted_class_id)
                    argmax_dict[k][j,np.argmax(probs)]=1
                    
                    if j==0:
                        probs_dict[k]=probs.reshape(1,-1)
                    else:
                        probs_dict[k]=np.append(probs_dict[k],probs.reshape(1,-1),axis=0)
                    
                #print("{}/{} paragraphs analyzed".format(j+1,len(paragraphs_filtered)))
                    
                j+=1
                
            m1_dict={}
            m2_dict={}
            m3_dict={}
            m4_dict={}
            
            header_lst=['Article length']
            preds_lst=[sum(para_lengths)]
            
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
                
            df_page=pd.DataFrame([preds_lst],columns=header_lst)
            
            if i==0:
                df_outputs=df_page
            else:
                df_outputs=pd.concat([df_outputs,df_page],axis=0)
            
            if show_progress==True:
                print("{}/{} articles analyzed".format(i+1,len(urls)))
            
        df_outputs=df_outputs.reset_index(drop=True)
        
        df_final=pd.concat([df_in,df_outputs],axis=1)
        
        if output_path!=None:
            os.chdir(output_path)
        
        df_final.to_csv(output_filename,index=False)
    