# Wikipedia-sentiment-analysis
Code for LLM-based sentiment analysis of Wikipedia articles

## Purpose
Wikipedia is the Internet's most commonly visited encyclopedia and is frequently used as an authoritative reference for a variety of subjects. One of Wikipedia's governing policies is neutral point of view (NPOV), which prohibits showing partiality or bias on the part of editors. For most articles on Wikipedia this is not a concern, but for articles regarding contentious topics or persons there is a greater chance that NPOV is violated. Using state-of-the-art large language models (LLMs) tailored for sentiment analysis can help to identify sentiments represented within Wikipedia articles, potentially identifying where NPOV may be at risk. 

## How it works
The LLMs used for this project belong to the Bidirectional encoder representations from transformers (BERT) family. These models tokenize input strings, and then embeddings are generated from the tokens via a transformer encoder:
<img width="1426" height="532" alt="image" src="https://github.com/user-attachments/assets/12b52a16-4e78-4ff9-a289-fd293fa32db6" />
In the case of sentiment analysis, the output of the transformer encoder undergoes classification to obtain a sentiment prediction:
<img width="1920" height="1392" alt="image" src="https://github.com/user-attachments/assets/78c729ea-95fc-482a-bfb5-4e8286d6aea7" />
