# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from pysentimiento import create_analyzer

def predict_sentiment(texts,tokenizer,model):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probabilities.detach().numpy()

class bert():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("MarieAngeA13/Sentiment-Analysis-BERT")
        self.model = AutoModelForSequenceClassification.from_pretrained("MarieAngeA13/Sentiment-Analysis-BERT")
    
    def predict(self,input_str):
        probabilities=predict_sentiment(input_str,self.tokenizer,self.model)
        return probabilities

class roberta():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    
    def predict(self,input_str):
        probabilities=predict_sentiment(input_str,self.tokenizer,self.model)
        return probabilities

class distilbert():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")
        self.model = AutoModelForSequenceClassification.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    
    def predict(self,input_str):
        probabilities=predict_sentiment(input_str,self.tokenizer,self.model)
        return np.flip(probabilities) #Flip to keep convention of 0=negative,1=neutral,2=positive

class robertuito():
    def __init__(self):
        self.analyzer = create_analyzer(task="sentiment", lang="en")
        
    def predict(self,input_str):
        result=self.analyzer.predict(input_str)
        probabilities=np.array([result.probas['NEG'],result.probas['NEU'],
                                result.probas['POS']])
        return probabilities
