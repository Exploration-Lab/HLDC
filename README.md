# HLDC

* We create a Hindi Legal Documents Corpus (HLDC) of 912,568  documents. These documents are cleaned and structured to make them usable for downstream NLP/IR applications. Moreover, this is a growing corpus as we continue to add more legal documents to HLDC. We release the corpus and model implementation code in this repo

* As a use-case for applicability of the corpus for developing legal systems, we propose the task of \emph{Bail Prediction}.

* For the task of bail prediction, we experiment with a variety of deep learning models. We propose a multi-task learning model based on transformer architecture. The proposed model uses extractive summarization as an auxiliary task and bail prediction as the main task.   


Each folder contains the different components of the paper namely collection, cleaning and information extraction, corpus analysis, summarisation models and bail prediction models. There are respective README's in each of the component where necessary to run the requisite code. 

## License

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

The HLDC dataset and software follows [CC-BY-NC](CC-BY-NC) license. Thus, users can share and adapt our dataset if they give credit to us and do not use our dataset for any commercial purposes.

