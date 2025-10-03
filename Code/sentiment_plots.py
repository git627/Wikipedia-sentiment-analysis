# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 11:21:50 2025

@author: marsh
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

def article_probability(models,df):
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

def plot_individual(models,input_filename,input_path=None,output_filename=
                    'figure.png',output_path=None,save_fig=True,plot_metric='paragraph_percent',
                    labels_col='Name'):

    sentiments=['Negative','Neutral','Positive']
    
    models=list(map(lambda x: model_name(x), models))
    
    df=pd.read_csv(input_filename)
    
    fig, axs = plt.subplots(1,len(models),figsize=(6*len(models), 5))
    
    if len(models)==1:
        axs=[axs]
    
    for i, ax in enumerate(axs):
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.tick_params(axis='y', labelsize=14, left=False)
    
    for i, col_label in enumerate(models):
        axs[i].set_title(col_label, fontsize=14)
    
    if input_path!=None:
        os.chdir(input_path)
    
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
    