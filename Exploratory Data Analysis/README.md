# Data Exploration
Here the code is to explore different dimensions of the HLDC corpus.

## GatheringStatistics.ipynb
It finds out statistics like number of cases per district, distribution of case types (Bail Applications, Criminal Cases, Civil Suits, etc.) across districts and also normalises case types.

## LexicalDiversity.ipynb
This script explores the linguistic features like number of words, sentences, unique words, etc.

## LexicalDiversity-TFIDF.ipynb
This script tried to find evidences of lexical diversity across geographical regions by using TFIDF where one document corresponds to all documents of a district.

## sample_read_zipped.py
This is a sample script to read the released data directly in the zip format without unzipping it also gives an idea about the internal structure of the released data.
