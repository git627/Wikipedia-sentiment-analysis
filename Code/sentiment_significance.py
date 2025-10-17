# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 09:59:15 2025

@author: marsh
"""

import pandas as pd
import numpy as np
from sentiment_helpers import model_name
from scipy.stats import mannwhitneyu
import sys
import os

def significance_test(models,input_filename,group_cols,group_values=None,metric='weighted_prob',
                      effect_size_metric=None,input_path=None,min_group_size=None):

    if input_path!=None:
        os.chdir(input_path)
    
    df=pd.read_csv(input_filename)
    
    if type(models)==str:
        models=[models]
        
    if type(group_values)==bool:
        group_values=[group_values]
        
    sig_metrics=['paragraph_percent','weighted_prob']
    effect_size_metrics=['f','r']
    
    if metric not in sig_metrics:
        raise ValueError('Please select metric from the following values: {}'.format(sig_metrics))
        sys.exit(1)
        
    if effect_size_metric!=None and effect_size_metric not in effect_size_metrics:
        raise ValueError('Please select effect size metric from the following values: {}'.format(sig_metrics))
        sys.exit(1)
        
    sentiments=['negative','neutral','positive']
    
    models_format=list(map(lambda x: model_name(x), models))
    
    combos_orig=df[group_cols]
    combos_filtered=combos_orig
        
    #If filter values unspecified, allow all values
    if group_values==None:
        group_values=[True]*len(group_cols)
    
    for i in range(len(group_cols)):
        if group_values[i]!=True:
            combos_filtered=combos_filtered[combos_filtered[group_cols[i]].isin(group_values[i])]
        
    combos_unique=combos_filtered.drop_duplicates()
    
    p_dict={}
    effect_dict={}
    
    for i in range(len(models)):
        p_dict[models[i]]={}
        effect_dict[models[i]]={}
        for j in range(len(sentiments)):
            if metric=='weighted_prob':
                columns=['Weighted probability of {} sentiment, {}'.format(sentiments[j],models_format[i])]
            elif metric=='paragraph_percent':
                columns=['Percent of paragraphs with {} sentiment, {}'.format(sentiments[j],models_format[i])]
            
            match_counts=[]
            labels=[]
            df_dict={}
            
            l=0
            
            for k in range(len(combos_unique)):
                target_row = combos_unique.iloc[k]
                matches_per_column = combos_orig == target_row
                all_columns_match = matches_per_column.all(axis=1)
                inds_match = combos_orig.index[all_columns_match].tolist()
                df_dict[k]=df.loc[inds_match]
                df_dict[k]=df_dict[k][columns]
                match_counts.append(len(inds_match))
                labels.append(', '.join(combos_unique.iloc[k].astype(str)))
                
                if min_group_size!=None:
                    if len(inds_match)<min_group_size:
                        del df_dict[k]
                        del labels[l]
                        l-=1
                        
                l+=1
            
            p_values=np.empty((len(df_dict.keys()),len(df_dict.keys())))
            effect_values=np.empty((len(df_dict.keys()),len(df_dict.keys())))
    
            for k in list(df_dict.keys()):  
                for l in list(df_dict.keys()):
                    U=mannwhitneyu(df_dict[k],df_dict[l])
                    k_lookup=list(df_dict.keys()).index(k)
                    l_lookup=list(df_dict.keys()).index(l)
                    p_values[k_lookup,l_lookup]=U[1][0]
                    U1=U[0]
                    n1=match_counts[k]
                    n2=match_counts[l]
                    f=U1/(n1*n2)
                    r=2*f-1
                    
                    if effect_size_metric==None or effect_size_metric=='f':
                        effect_values[k_lookup,l_lookup]=f[0]
                    elif effect_size_metric=='r':
                        effect_values[k_lookup,l_lookup]=r[0]
            
            p_df=pd.DataFrame(p_values,index=labels,columns=labels)
            effect_df=pd.DataFrame(effect_values,index=labels,columns=labels)
            p_dict[models[i]][sentiments[j]]=p_df
            effect_dict[models[i]][sentiments[j]]=effect_df
            
    if effect_size_metric==None:
        return p_dict
    else:
        return p_dict,effect_dict
