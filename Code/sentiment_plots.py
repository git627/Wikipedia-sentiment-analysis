# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 11:21:50 2025

@author: marsh
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dcor
from sentiment_helpers import model_name

def h_stack_bar_chart(fig, ax, column, round_digits, results, labels, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlGn'](
        np.linspace(0.15, 0.85, data.shape[1]))

    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        text_color = 'black'
        ax.bar_label(rects, fmt='%.{}f'.format(round_digits), label_type='center', color=text_color,fontsize=12,fontweight='bold')
        if column>0:
            ax.set_yticks([])

    return fig, ax

def paragraph_percent(models,df):
    title_str='Percent of paragraphs by sentiment'
    round_digits=1
    
    plot_dict={}
    
    for i in range(len(models)):
        plot_dict[models[i]]={}
        for j in range(len(df)):
            pcts=np.array([df.loc[j,'Percent of paragraphs with negative sentiment, {}'.format(models[i])],
                       df.loc[j,'Percent of paragraphs with neutral sentiment, {}'.format(models[i])],
                       df.loc[j,'Percent of paragraphs with positive sentiment, {}'.format(models[i])]])
            plot_dict[models[i]][j]=pcts
            
    return title_str,round_digits,plot_dict

def avg_paragraph_percent(models,df_dict):
    title_str='Average percent of paragraphs by sentiment'
    round_digits=1
    
    plot_dict={}
    
    for i in range(len(models)):
        plot_dict[models[i]]={}
        for j in list(df_dict.keys()):
            counts=np.array(df_dict[j][['Number of paragraphs with negative sentiment, {}'.format(models[i]),
                       'Number of paragraphs with neutral sentiment, {}'.format(models[i]),
                       'Number of paragraphs with positive sentiment, {}'.format(models[i])]].sum(axis=0))
            plot_dict[models[i]][j]=100*counts/np.sum(counts)
            
    return title_str,round_digits,plot_dict

def article_prob(models,df):
    title_str='Article sentiment probability'
    round_digits=2
    
    plot_dict={}
    
    for i in range(len(models)):
        plot_dict[models[i]]={}
        for j in range(len(df)):
            probs=np.array([df.loc[j,'Weighted probability of negative sentiment, {}'.format(models[i])],
                       df.loc[j,'Weighted probability of neutral sentiment, {}'.format(models[i])],
                       df.loc[j,'Weighted probability of positive sentiment, {}'.format(models[i])]])
            plot_dict[models[i]][j]=probs
            
    return title_str,round_digits,plot_dict

def avg_article_prob(models,df_dict):
    title_str='Average article sentiment probability'
    round_digits=2
    
    plot_dict={}
    
    for i in range(len(models)):
        plot_dict[models[i]]={}
        for j in list(df_dict.keys()):
            prob_avg=np.array(df_dict[j][['Weighted probability of negative sentiment, {}'.format(models[i]),
                       'Weighted probability of neutral sentiment, {}'.format(models[i]),
                       'Weighted probability of positive sentiment, {}'.format(models[i])]].mean(axis=0))
            plot_dict[models[i]][j]=prob_avg
        
    return title_str,round_digits,plot_dict

def percent_most_common(models,df_dict):
    title_str='Percent of articles by most common sentiment'
    round_digits=1
    
    plot_dict={}
    
    for i in range(len(models)):
        plot_dict[models[i]]={}
        for j in list(df_dict.keys()):
            counts=np.array([df_dict[j]['Most common sentiment, {}'.format(models[i])].str.count('Negative').sum(),
            df_dict[j]['Most common sentiment, {}'.format(models[i])].str.count('Neutral').sum(),
            df_dict[j]['Most common sentiment, {}'.format(models[i])].str.count('Positive').sum()])
            plot_dict[models[i]][j]=100*counts/np.sum(counts)
            
    return title_str,round_digits,plot_dict
        
def percent_highest_weighted_vote(models,df_dict):
    title_str='Percent of articles by sentiment with the highest weighted vote'
    round_digits=1
    
    plot_dict={}
    
    for i in range(len(models)):
        plot_dict[models[i]]={}
        for j in list(df_dict.keys()):
            counts=np.array([df_dict[j]['Sentiment with highest weighted vote, {}'.format(models[i])].str.count('Negative').sum(),
            df_dict[j]['Sentiment with highest weighted vote, {}'.format(models[i])].str.count('Neutral').sum(),
            df_dict[j]['Sentiment with highest weighted vote, {}'.format(models[i])].str.count('Positive').sum()])
            plot_dict[models[i]][j]=100*counts/np.sum(counts)
            
    return title_str,round_digits,plot_dict
        
def percent_highest_weighted_prob(models,df_dict):
    title_str='Percent of articles by sentiment with the highest weighted probability'
    round_digits=1
    
    plot_dict={}
    
    for i in range(len(models)):
        plot_dict[models[i]]={}
        for j in list(df_dict.keys()):
            counts=np.array([df_dict[j]['Sentiment with highest weighted probability, {}'.format(models[i])].str.count('Negative').sum(),
            df_dict[j]['Sentiment with highest weighted probability, {}'.format(models[i])].str.count('Neutral').sum(),
            df_dict[j]['Sentiment with highest weighted probability, {}'.format(models[i])].str.count('Positive').sum()])
            plot_dict[models[i]][j]=100*counts/np.sum(counts)
        
    return title_str,round_digits,plot_dict

def plot_individual(models,input_filename,input_path=None,output_filename=
                    'figure.png',output_path=None,save_fig=True,plot_metric='paragraph_percent',
                    labels_col='Name'):

    sentiments=['Negative','Neutral','Positive']
    
    if type(models)==str:
        models=[models]
    
    models=list(map(lambda x: model_name(x), models))
    
    if input_path!=None:
        os.chdir(input_path)
    
    df=pd.read_csv(input_filename)
    
    individual_metrics=['paragraph_percent','article_prob']

    if plot_metric not in individual_metrics:
        raise ValueError('Please select plot_metric from the following values: {}'.format(individual_metrics))
        sys.exit(1)
    
    fig, axs = plt.subplots(1,len(models),figsize=(6*len(models), 5))
    
    if len(models)==1:
        axs=[axs]
    
    for i, ax in enumerate(axs):
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.tick_params(axis='y', labelsize=14, left=False)
    
    for i, col_label in enumerate(models):
        axs[i].set_title(col_label, fontsize=14)
    
    labels=df[labels_col]
    
    title_str,round_digits,plot_dict=globals()[plot_metric](models,df)
    
    for i in range(len(models)):
        h_stack_bar_chart(fig,axs[i],i,round_digits,plot_dict[models[i]], labels, sentiments)
        
    x_coord=-0.59*len(models)+1.08
    
    ax.legend(ncols=len(sentiments), loc='lower center', bbox_to_anchor=(x_coord, -0.15), fontsize='large')
    fig.suptitle(title_str, fontsize=20,y=1.05)
    
    if output_path!=None:
        os.chdir(output_path)
    
    if save_fig==True:
        plt.savefig(output_filename, bbox_inches = 'tight')
        
def plot_group(models,input_filename,label_cols,input_path=None,output_filename='figure.png',
               output_path=None,save_fig=True,plot_metric='avg_paragraph_percent',
               label_values=None,min_group_size=None):
     
    sentiments=['Negative','Neutral','Positive']
    
    if type(models)==str:
        models=[models]
    
    models=list(map(lambda x: model_name(x), models))
    
    if input_path!=None:
        os.chdir(input_path)
    
    df=pd.read_csv(input_filename)
    
    combos_orig=df[label_cols]
    
    combos_filtered=combos_orig
    
    #If filter values unspecified, allow all values
    if label_values==None:
        label_values=[True]*len(label_cols)
    
    for i in range(len(label_cols)):
        if label_values[i]!=True:
            combos_filtered=combos_filtered[combos_filtered[label_cols[i]].isin(label_values[i])]
        
    combos_unique=combos_filtered.drop_duplicates()
    match_counts=[]
    labels=[]
    df_dict={}
    
    k=0
        
    for i in range(len(combos_unique)):
        target_row = combos_unique.iloc[i]
        matches_per_column = combos_orig == target_row
        all_columns_match = matches_per_column.all(axis=1)
        inds_match = combos_orig.index[all_columns_match].tolist()
        df_dict[i]=df.loc[inds_match]
        match_counts.append(len(inds_match))
        labels.append(', '.join(combos_unique.iloc[i].astype(str))+' (n = {})'.format(match_counts[i]))
        
        if min_group_size!=None:
            if len(inds_match)<min_group_size:
                del df_dict[i]
                del labels[k]
                k-=1
                
        k+=1
        
    group_metrics=['avg_paragraph_percent','percent_most_common','percent_highest_weighted_vote',
                   'percent_highest_weighted_prob','avg_article_prob']
    
    if plot_metric not in group_metrics:
        raise ValueError('Please select plot_metric from the following values: {}'.format(group_metrics))
        sys.exit(1)
        
    fig, axs = plt.subplots(1,len(models),figsize=(6*len(models), 5))
    
    if len(models)==1:
        axs=[axs]
    
    for i, ax in enumerate(axs):
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.tick_params(axis='y', labelsize=14, left=False)
    
    for i, col_label in enumerate(models):
        axs[i].set_title(col_label, fontsize=14)
    
    title_str,round_digits,plot_dict=globals()[plot_metric](models,df_dict)
    
    for i in range(len(models)):
        h_stack_bar_chart(fig,axs[i],i,round_digits,plot_dict[models[i]], labels, sentiments)
        
    x_coord=-0.59*len(models)+1.08
    
    ax.legend(ncols=len(sentiments), loc='lower center', bbox_to_anchor=(x_coord, -0.15), fontsize='large')
    fig.suptitle(title_str, fontsize=20,y=1.05)
    
    if output_path!=None:
        os.chdir(output_path)
    
    if save_fig==True:
        plt.savefig(output_filename, bbox_inches = 'tight')
    
def plot_corrs(models,input_filename,x_col,input_path=None,output_filename='figure.png',
              output_path=None,save_fig=True,plot_metric='paragraph_percent',
              filter_cols=None,filter_values=None):

    sentiments=['Negative','Neutral','Positive']
    
    if type(models)==str:
        models=[models]
    
    models=list(map(lambda x: model_name(x), models))
    
    if input_path!=None:
        os.chdir(input_path)
    
    df=pd.read_csv(input_filename)
    
    if filter_cols!=None:
    
        combos_orig=df[filter_cols]
        combos_filtered=combos_orig
        
        for i in range(len(filter_cols)):
            combos_filtered=combos_filtered[combos_filtered[filter_cols[i]].isin(filter_values[i])]
                
        inds_match=combos_filtered.index
        df=df.loc[inds_match]        
        df=df.reset_index(drop=True)
    
    x_vals=df[x_col].to_numpy()
    
    corr_metrics=['paragraph_percent','article_prob']
    
    if plot_metric not in corr_metrics:
        raise ValueError('Please select plot_metric from the following values: {}'.format(corr_metrics))
        sys.exit(1)
        
    title_str,_,plot_dict=globals()[plot_metric](models,df)
    
    plot_array_dict={}
    
    for i in range(len(models)):
        plot_array_dict[models[i]]=np.empty((df.shape[0],3))
        for j in range(df.shape[0]):
            plot_array_dict[models[i]][j]=plot_dict[models[i]][j]
            
    fig, axs = plt.subplots(nrows=len(models),ncols=3,figsize=(18, 4*len(models)))
        
    corrs=np.empty((len(models),3))
    dcorrs=np.empty((len(models),3))
    
    if min(x_vals)>=0:
        min_x_plot=0.95*min(x_vals)
    else:
        min_x_plot=1.05*min(x_vals)
        
    if max(x_vals)>=0:
        max_x_plot=1.05*max(x_vals)
    else:
        max_x_plot=0.95*max(x_vals)
        
    if max_x_plot<=min_x_plot:
        if min_x_plot>=0:
            max_x_plot=1.05*min_x_plot
        else:
            max_x_plot=0.95*min_x_plot
            
    if plot_metric=='paragraph_percent':
        y_lim_max=100
    elif plot_metric=='article_prob':
        y_lim_max=1
            
    y_coord=-0.06*len(models)+1.16
        
    for i in range(len(models)):
        for j in range(3):
            count_low=len(np.where(plot_array_dict[models[i]][:,j]<=0.5*y_lim_max)[0])
            count_high=len(np.where(plot_array_dict[models[i]][:,j]>0.5*y_lim_max)[0])
            if count_high>count_low:
                text_height=0.15
            else:
                text_height=0.85
            if len(models)>1:
                corrs[i,j]=np.corrcoef(x_vals,plot_array_dict[models[i]][:,j])[0,1] #Pearson correlations
                dcorrs[i,j]=dcor.distance_correlation(x_vals.astype(np.float64),plot_array_dict[models[i]][:,j].astype(np.float64)) #distance correlations
                axs[i,j].plot(x_vals,plot_array_dict[models[i]][:,j],'b.')
                axs[i,j].set_xlim(min_x_plot,max_x_plot)
                axs[i,j].set_ylim(0,y_lim_max)
                axs[i,j].text(0.15*(max(x_vals)-min(x_vals))+min(x_vals),text_height*y_lim_max,r'$\rho$ = {}, dCor = {}'.format(np.round(corrs[i,j],2),np.round(dcorrs[i,j],2)),
                              fontsize=18)
                axs[i,j].tick_params(axis='x', labelsize=14)
                axs[i,j].tick_params(axis='y', labelsize=14)
                if i==0:
                    axs[i,j].set_title(sentiments[j], fontsize=22)
                if j==0:
                    axs[i,j].set_ylabel(models[i],fontsize=22)
            else:
                corrs[i,j]=np.corrcoef(x_vals,plot_array_dict[models[i]][:,j])[0,1] #Pearson correlations
                dcorrs[i,j]=dcor.distance_correlation(x_vals.astype(np.float64),plot_array_dict[models[i]][:,j].astype(np.float64)) #distance correlations
                axs[j].plot(x_vals,plot_array_dict[models[i]][:,j],'b.')
                axs[j].set_xlim(min_x_plot,max_x_plot)
                axs[j].set_ylim(0,y_lim_max)
                axs[j].text(0.15*(max(x_vals)-min(x_vals))+min(x_vals),text_height*y_lim_max,r'$\rho$ = {}, dCor = {}'.format(np.round(corrs[i,j],2),np.round(dcorrs[i,j],2)),
                              fontsize=18)
                axs[j].tick_params(axis='x', labelsize=14)
                axs[j].tick_params(axis='y', labelsize=14)
                if i==0:
                    axs[j].set_title(sentiments[j], fontsize=22)
                if j==0:
                    axs[j].set_ylabel(models[i],fontsize=22)
    
    fig.suptitle('{} vs. {}, n = {}'.format(title_str,x_col,len(df)), fontsize=28,y=y_coord)
        
    if output_path!=None:
        os.chdir(output_path)
    
    if save_fig==True:
        plt.savefig(output_filename, bbox_inches = 'tight')