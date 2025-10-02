# Wikipedia-sentiment-analysis
Code for LLM-based sentiment analysis of Wikipedia articles

## Purpose
Wikipedia is the Internet's most commonly visited encyclopedia and is frequently used as an authoritative reference for a variety of subjects. One of Wikipedia's governing policies is neutral point of view (NPOV), which prohibits showing partiality or bias on the part of editors. For most articles on Wikipedia this is not a concern, but for articles regarding contentious topics or persons there is a greater chance that NPOV is violated. Using state-of-the-art large language models (LLMs) tailored for sentiment analysis can help to identify sentiments represented within Wikipedia articles, potentially identifying where NPOV may be at risk. There is some early research into this topic[^1], but it is nonetheless a very new field of study.

## How it works
The LLMs used for this project belong to the Bidirectional encoder representations from transformers (BERT)[^2] family. These models tokenize input strings, and then embeddings are generated from the tokens via a transformer encoder:

<p align="center">
<img width="809" height="383" alt="image" src="https://github.com/user-attachments/assets/9db027a3-7604-4ebe-85cb-9921c3bd2f23" />
</p>

In the case of sentiment analysis, the output of the transformer encoder undergoes classification to obtain a sentiment prediction:

<p align="center">
<img width="439" height="323" alt="image" src="https://github.com/user-attachments/assets/b7e07b04-a2c6-4298-bbee-844270903d4c" />
</p>

The classification of the sentiment can be binary (i.e., positive or negative) or multi-class (e.g., positive, negative, or neutral). The probabilities for each prediction label are also returned. For this project a negative/neutral/positive classification scheme was used. 

## Code
The code works by scraping Wikipedia articles via an API and dividing the articles into paragraphs. Limiting the length of input strings into the models is necessary since there is a token length limit of 512. The paragraphs are then fed into the BERT models and classified according to sentiment. Sentiment analysis at the article level can be performed by aggregating the sentiments of their constituent paragraphs. For this project, three aggregrating methods are used:

1. Most common: the sentiment of the article is selected as the most-represented sentiment among all the paragraphs. Additionally, a percent breakdown of paragraphs by sentiment is provided.
2. Greatest sum of weighted votes: similar to the most common method, but the votes are multiplied by the lengths of their corresponding paragraphs
3. Highest weighted probability: the prediction level probabilities are aggregated across all paragraphs according to:
<p align="center">
<img width="346" height="49" alt="image" src="https://github.com/user-attachments/assets/b7f2472c-828b-4730-9396-648c1f696b8b" />
</p>

Where <img width="28" height="23" alt="image" src="https://github.com/user-attachments/assets/b0f2e562-f184-45d1-87fe-dc8c64e2efee" /> is the aggregated probability of the jth sentiment class, <img width="14" height="17" alt="image" src="https://github.com/user-attachments/assets/f928cbfd-150e-4351-9940-41539e99ad4e" /> is the length of the ith text segment in characters, <img width="28" height="16" alt="image" src="https://github.com/user-attachments/assets/8c37f7fc-a872-4cff-8fa5-a35b381d2c3d" /> is the probability of the ith text segment belonging to the jth sentiment class, and <img width="13" height="15" alt="image" src="https://github.com/user-attachments/assets/a1e4b1e1-e6d5-4917-8717-dbe8a7114cdc" /> is the total number of text segments within the article.

## Files
There are currently four scripts used for this project:

1. wiki_sentiment_single.py: this script allows users to generate sentiment predictions for a single Wikipedia article. The user provides a URL as input.
2. wiki_sentiment_multi.py: this script allows users to generate sentiment predictions for multiple Wikipedia articles. The user provides either a list of URLs or a UTF-8 encoded .csv file as input.
3. sentiment_models.py: this script contains classes representing the various BERT models used for this work. Users are free to augment it as they see fit. Currently, there is one basic BERT model, one RoBERTa[^3] model, one DistilBERT[^4] model, and the RoBERTuito[^5] model found in the pysentimiento[^6] package.
4. sentiment_helpers.py: this script contains various helper functions used by the other scripts.

Additionally, two Jupyter notebooks (wiki_single_example.ipynb and wiki_multi_example.ipynb) are provided as examples for anyone who would like to see how the scripts are used.

[^1]:Stróżyna, Milena, et al. "Sentiment Analysis of Wikipedia Articles About Companies: A Comparison of Different Models." International Conference on Business Information Systems. Cham: Springer Nature Switzerland, 2025.
[^2]:Devlin, Jacob; Chang, Ming-Wei; Lee, Kenton; Toutanova, Kristina (October 11, 2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding".
[^3]:Liu, Yinhan; Ott, Myle; Goyal, Naman; Du, Jingfei; Joshi, Mandar; Chen, Danqi; Levy, Omer; Lewis, Mike; Zettlemoyer, Luke; Stoyanov, Veselin (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach".
[^4]:Sanh, Victor; Debut, Lysandre; Chaumond, Julien; Wolf, Thomas (February 29, 2020), "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter".
[^5]:Pérez, Juan Manuel, et al. "RoBERTuito: a pre-trained language model for social media text in Spanish." arXiv preprint arXiv:2111.09453 (2021).
[^6]:Pérez, Juan Manuel, et al. "pysentimiento: A python toolkit for opinion mining and social nlp tasks." arXiv preprint arXiv:2106.09462 (2021).

