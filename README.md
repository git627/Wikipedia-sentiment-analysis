# Wikipedia-sentiment-analysis
Code for LLM-based sentiment analysis of Wikipedia articles

## Purpose
Wikipedia is the Internet's most commonly visited encyclopedia and is frequently used as an authoritative reference for a variety of subjects. One of Wikipedia's governing policies is neutral point of view (NPOV), which prohibits showing partiality or bias on the part of editors. For most articles on Wikipedia this is not a concern, but for articles regarding contentious topics or persons there is a greater chance that NPOV is violated. Using state-of-the-art large language models (LLMs) tailored for sentiment analysis can help to identify sentiments represented within Wikipedia articles, potentially identifying where NPOV may be at risk. There is some early research into this topic (please see Sentiment Analysis of Wikipedia Articles About Companies: A Comparison of Different Models by Stróżyna et al. 2025). Nevertheless, the work in this area is still very new.

## How it works
The LLMs used for this project belong to the Bidirectional encoder representations from transformers (BERT) family. These models tokenize input strings, and then embeddings are generated from the tokens via a transformer encoder:

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

1. Most common: the sentiment of the article is selected as the most-represented sentiment among all the paragraphs. Additionally, a percent breakdown of paragraphs by sentiment may be provided.
2. Greatest sum of weighted votes: similar to the most common method, but the votes are multiplied by the lengths of their corresponding paragraphs
3. Highest weighted probability: the prediction level probabilities are aggregated across all paragraphs according to:
<p align="center">
<img width="346" height="49" alt="image" src="https://github.com/user-attachments/assets/b7f2472c-828b-4730-9396-648c1f696b8b" />
</p>

Where <img width="28" height="23" alt="image" src="https://github.com/user-attachments/assets/b0f2e562-f184-45d1-87fe-dc8c64e2efee" /> is the aggregated probability of the jth sentiment class, <img width="14" height="17" alt="image" src="https://github.com/user-attachments/assets/f928cbfd-150e-4351-9940-41539e99ad4e" /> is the length of the ith text segment in characters, <img width="28" height="16" alt="image" src="https://github.com/user-attachments/assets/8c37f7fc-a872-4cff-8fa5-a35b381d2c3d" /> is the probability of the ith text segment belonging to the jth sentiment class, and <img width="13" height="15" alt="image" src="https://github.com/user-attachments/assets/a1e4b1e1-e6d5-4917-8717-dbe8a7114cdc" /> is the total number of text segments within the article.
