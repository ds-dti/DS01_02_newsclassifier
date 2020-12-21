# Automated News Categorization and Sentiment Analysis using Indonesian Bidirectional Encoder Representations from Transformers (IndoBERT) for Technology, Health, and Business Analytics.

![Poster Tubes DTI](https://user-images.githubusercontent.com/54322914/102610527-ee430400-415f-11eb-9a4c-f9300d2a11f6.png)

This reepository is final project which mandatory by Digital Talent Incubator DTI 2020. The purpose of this project is help news anchor or pimred classify their news and know the sentiment of their news

Explanation Video: https://youtu.be/7iDCL7PgnOA

## About

To make Sustainable Development Goals (SDG) become stronger. Digital Talent Incubator (DTI) which handled by Telkom Indonesia give us project that can be used in the future and solve SDG problems. Our team consists of talented people want to solve problem in SDG 16.

Teams :
- Apriandito (Lecturer)
- Muhammad Shabri Arrahim Mardi (Team Leader)
- Ahmad Shohibus Sulthoni
- Farah Qotrunnada

## Background

SDG 16 : **Promote peaceful and inclusive societies for sustainable development, provide access to justice for all and build effective, accountable and inclusive institutions at all levels**

This project is build to help News Anchor, News Director, News Writer evaluate their news using Sentiment Classification. Why they need to classify it? Because for peaceful and inclusive societies, News play a major role in leading public opinion. So, news should be contain positive sentiment or neutral. 

Meanwhile, they also need to categorize or classify their news using exact class. For example, the news publisher has 3 categories which are Technology, Business and Health. It would be dificult if the news writer classify manually their news. So, this classifier hopefully can solve and help them.

### IndoBERT
To know more about BERT we recommend you to read this article [here](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270). Big thanks for all researcher who built pre-trained model IndoBERT. IndoBERT was built as part of IndoNLU. 


>IndoNLU is a collection of Natural Language Understanding (NLU) resources for Bahasa Indonesia with 12 downstream tasks. We provide the code to reproduce the results and large pre-trained models (IndoBERT and IndoBERT-lite) trained with around 4 billion word corpus (Indo4B), more than 20 GB of text data. This project was initially started by a joint collaboration between universities and industry, such as Institut Teknologi Bandung, Universitas Multimedia Nusantara, The Hong Kong University of Science and Technology, Universitas Indonesia, Gojek, and Prosa.AI.

For detail documentation and code you can go [here](https://github.com/indobenchmark/indonlu)

## How to Use

Below is explaination to use or modify this project.

### Directories Explaination
&nbsp;--dataset/ </br>
&nbsp;&nbsp;--news_category/ </br>
&nbsp;&nbsp;--smsa_doc-sentiment-prosa/ </br>
&nbsp;--model/ </br>
&nbsp;--notebook/ </br>
&nbsp;--utils/ </br>
&nbsp;classifier.py </br>

- dataset folder contains news and sentiment dataset. For the sentiment dataset, we use dataset from indoNLU. But for the news category we use Kompas.com news index. You can change with your own dataset.
- model folder contains 2 binary file which are model for sentiment analysis and model for category classification.
- Notebook folder contains ipynb files for tuning the pre-trained model. You can experiment with your own.
- utils folder contains py file which used in jupyter notebook. All utils are inspired by IndoNLU.
- classifier.py is python script that used to inferencing model.

footnote : **We train the model using GPU. If you don't have any GPU, you may retrain the model**

### Prerequisites

Install these library :
- pandas
- pytorch
- transformers
- sklearn
- re

Then run your notebook or classifier.py </br>
`python classifier.py`

## API Docs 

Here's a sample request to the API:

```bash
curl -X POST "http://127.0.0.1:8000/predict/" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"text\":\"Menteri Ketenagakerjaan Ida Fauziyah mengatakan, untuk melindungi   Pekerja  Migran Indonesia (PMI), kuncinya adalah sinergitas dan kolaborasi seluruh pihak.\"}"
```

or, if you using python:

```sh
import requests

myText = "{\"text\":\"Menteri Ketenagakerjaan Ida Fauziyah mengatakan, untuk melindungi   Pekerja  Migran Indonesia (PMI), kuncinya adalah sinergitas dan kolaborasi seluruh pihak.\"}"
url = "http://127.0.0.1:8000/predict/"
response = requests.post(url, data=myText)
print(response.json())
```

The response you'll get looks something like this:

```js
{
  "text": "Menteri Ketenagakerjaan Ida Fauziyah mengatakan, untuk melindungi   Pekerja  Migran Indonesia (PMI), kuncinya adalah sinergitas dan kolaborasi seluruh pihak.",
  "prediction": {
    "category": [
      {
        "label": "business",
        "conf": "87.66"
      }
    ],
    "sentiment": [
      {
        "label": "neutral",
        "conf": "99.34"
      }
    ]
  }
}
```

## The backbone of our REST API will be:

FastAPI - lets you easily set up a REST API (some say it might be fast, too)

Uvicorn - server that lets you do async programming with Python (pretty cool)

Pydantic - data validation by introducing types for our request and response data.



## Installation

Clone this repo:

```sh
git clone https://github.com/shabri-arrahim/DS01_02_newsclassifier.git
cd DS01_02_newsclassifier
```

Install the prerequisites:

```sh
pip install requirements.txt
```

Run in local:
```sh
cd app
python app.py
```
