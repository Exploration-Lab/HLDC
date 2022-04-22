# HLDC

* We create a Hindi Legal Documents Corpus (HLDC) of 912,568  documents. These documents are cleaned and structured to make them usable for downstream NLP/IR applications. Moreover, this is a growing corpus as we continue to add more legal documents to HLDC. We release the corpus and model implementation code in this repo

* As a use-case for applicability of the corpus for developing legal systems, we propose the task of Bail Prediction.

* For the task of bail prediction, we experiment with a variety of deep learning models. We propose a multi-task learning model based on transformer architecture. The proposed model uses extractive summarization as an auxiliary task and bail prediction as the main task.   


Each folder contains the different components of the paper namely collection, cleaning and information extraction, corpus analysis, summarisation models and bail prediction models. There are respective README's in each of the component where necessary to run the requisite code. 
## Dataset
Please fill the following [form](https://docs.google.com/forms/d/e/1FAIpQLSf6c8fuWWB-VWf4Yv5YPd9dkfE7bnX8y7hDt0DKCKfW_ocDBQ/viewform) to get access to the HLDC dataset.
Please note the dataset is only to be used for research purposes.

## License

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

The HLDC dataset and software follows [CC-BY-NC](CC-BY-NC) license. Thus, users can share and adapt our dataset if they give credit to us and do not use our dataset for any commercial purposes.

### Paper: https://arxiv.org/abs/2204.00806 

## Citation

```
@inproceedings{kapoor-etal-2022-HLDC,
    title = "{HLDC}: Hindi Legal Documents Corpus",
    author = "Kapoor, Arnav and 
              Dhawan, Mudit and
              Goel, Anmol and 
              Arjun, T.H and 
              Agrawal, Vibhu and 
              Agrawal, Amul and
              Bhattacharya, Arnab and 
              Kumaraguru, Ponnurangam and
              Modi, Ashutosh",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2022",
    year = "2022",
    publisher = "Association for Computational Linguistics",
    abstract = "Many populous countries including India are burdened with a considerable backlog of legal cases. Development of automated systems that could process legal documents and augment legal practitioners can mitigate this. However, there is a dearth of high-quality corpora that is needed to develop such data-driven systems. The problem gets even more pronounced in the case of low resource languages such as Hindi. In this resource paper, we introduce the Hindi Legal Documents Corpus (HLDC), a corpus of more than 900K legal documents in Hindi. The documents are cleaned and structured to enable the development of downstream applications. Further, as a use-case for the corpus, we introduce the task of Bail Prediction. We experiment with a battery of models and propose a multi-task learning (MTL) based model for the same. MTL models use summarization as an auxiliary task along with bail prediction as the main task. Results on different models are indicative of the need for further research in this area. 
",
}
```
