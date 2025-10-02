# Wikipedia-sentiment-analysis
Code for LLM-based sentiment analysis of Wikipedia articles

## Purpose
Wikipedia is the Internet's most commonly visited encyclopedia and is frequently used as an authoritative reference for a variety of subjects. One of Wikipedia's governing policies is neutral point of view (NPOV), which prohibits showing partiality or bias on the part of editors. For most articles on Wikipedia this is not a concern, but for articles regarding contentious topics or persons there is a greater chance that NPOV is violated. Using state-of-the-art large language models (LLMs) tailored for sentiment analysis can help to identify sentiments represented within Wikipedia articles, potentially identifying where NPOV may be at risk. 

## How it works
The LLMs used for this project belong to the Bidirectional encoder representations from transformers (BERT) family. These models tokenize input strings, and then embeddings are generated from the tokens via a transformer encoder:

<p align="center">
<img width="809" height="383" alt="image" src="https://github.com/user-attachments/assets/9db027a3-7604-4ebe-85cb-9921c3bd2f23" />
</p>

In the case of sentiment analysis, the output of the transformer encoder undergoes classification to obtain a sentiment prediction:

<p align="center">
<img width="439" height="323" alt="image" src="https://github.com/user-attachments/assets/b7e07b04-a2c6-4298-bbee-844270903d4c" />
</p>
