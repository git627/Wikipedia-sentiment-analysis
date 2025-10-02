# -*- coding: utf-8 -*-

import wikipediaapi
import numpy as np

def get_wikipedia_paragraphs_with_api(url):
    page_title= url[30:len(url)]
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent='MyWikipediaApp (your-email@example.com)', 
        language='en'
    )
    page_py = wiki_wiki.page(page_title)

    if page_py.exists():
        #return page_py.title, page_py.text

        paragraphs = []
        # Get the full text of the article
        full_text = page_py.text
    
        # Split the full text into paragraphs based on newlines
        # This is a common heuristic for paragraph separation in plain text
        raw_paragraphs = full_text.split('\n')
    
        for para in raw_paragraphs:
            stripped_para = para.strip()
            if stripped_para: # Only add non-empty paragraphs
                paragraphs.append(stripped_para)
                
        return paragraphs
            
    else:
        print(f"Page '{page_title}' does not exist.")
        return None, None

def split_by_closest_period(text):
    """
    Splits a string into two halves at the period closest to the middle.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    period_indices = [i for i, char in enumerate(text) if char == '.']

    if not period_indices:
        # If no periods are found, split at the exact middle
        mid_index = len(text) // 2
        return text[:mid_index], text[mid_index:]

    middle_of_string = len(text) / 2
    closest_period_index = min(period_indices, key=lambda x: abs(x - middle_of_string))

    # Split the string at the closest period.
    # The period itself can be included in the first or second half,
    # depending on preference. Here, it's included in the first half.
    first_half = text[:closest_period_index + 1]
    second_half = text[closest_period_index + 1:]

    return first_half, second_half

def id_to_sentiment(id):
    sentiments=['Negative','Neutral','Positive']
    return sentiments[id]

def Method1(sentiments):
    #Takes list of sentiments and returns the most common one
    pred=max(set(sentiments), key=sentiments.count)
    return pred

def Method2(argmax_array,para_lengths):
    #Takes array of votes as well as list of input string lengths and returns the sentiment with the
    #greatest sum of weighted votes
    pred=id_to_sentiment(np.argmax(np.sum(argmax_array*np.array(para_lengths).reshape(-1,1),axis=0)))
    return pred

def Method3(prob_array,para_lengths):
    #Takes array of probabilities as well as list of input string lengths and returns the sentiment
    #with the highest weighted probability as well as the weighted probabilities. Order for sentiment
    #probabilities is Negative, Neutral, Positive
    probs=np.sum(prob_array*np.array(para_lengths).reshape(-1,1),axis=0)/sum(para_lengths)
    pred=id_to_sentiment(np.argmax(probs))
    probs=probs.reshape(1,-1)
    
    return pred,probs

def Method4(argmax_array):
    #Takes array of votes and returns count and percent of votes corresponding to each sentiment. 
    #Order of sentiments is Negative, Neutral, Positive
    totals=np.sum(argmax_array,axis=0)
    pred=(totals/np.sum(totals)).reshape(1,-1)
    return totals.reshape(1,-1),pred