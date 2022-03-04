## Running Order
1. [OCR](./ocr.py)
2. [NER_and_document_division.ipynb](./NER_and_Document_division.ipynb)
3. [bail_decision_and_amount_extraction](./bail_decision_and_amount_extraction.ipynb)
4. [RemovingJudgeOpinion](./RemovingJudgeOpinion.ipynb)
5. [train_test_split_code](./train_test_split_code.ipynb)

## Brief Overview
### OCR
- Uses tesseract to convert raw PDFs downloaded in the data collection step to txt files. 
- Uses intermediatary Ghostscript to convert pdf to tiff files.
### NER_and_Document_division
- Removes Invalid Documents.
- Divides valid OCRed bail documents into `header`, `body` and `result` sections.
- Does NER removal on all Document.
### bail_decision_and_amount_extraction
- Extracts bail decision and amount from the documents.
### RemovingJudgeOpinion
- Divides body into `facts-and-arguments` and `judge-opinion` sections.
### train_test_split_code
- getting the train-test split.
