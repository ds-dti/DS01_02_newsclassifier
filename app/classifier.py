import os
import torch
import torch.nn.functional as F
import pandas as pd

from transformers import BertForSequenceClassification, BertConfig
from transformers import BertTokenizer

class SentimentClassifier():
    
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
        config = BertConfig.from_pretrained("indobenchmark/indobert-base-p1", num_labels=3)
        self.model = BertForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", config=config)
        self.model.load_state_dict(torch.load(os.path.join("../model", "model_sentiment.bin"), map_location='cpu'))
        self.results = list()

    def predict(self,text):
        i2w = {0: 'positive', 1: 'neutral', 2: 'negative'}
        text = text[:512]
        subwords = self.tokenizer.encode(text)
        subwords = torch.LongTensor(subwords).view(1,-1).to(self.model.device)

        logits = self.model(subwords)[0]
        label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

        return i2w[label] , str(f'{F.softmax(logits, dim=-1).squeeze()[label] * 100:.2f}')
    
    def process_text(self,text):
        result = dict()
        label , conf = self.predict(text=text)
        print(f"Success Predict {label}, {conf}")
        # result['text'] = text
        result['label'] = label
        result['conf'] = conf
        self.results.append(result)
        # print(self.results)
    
    def get_result(self):
        return self.results
    def reset(self):
        self.results = list()

class CategoryClassifier():
    
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
        config = BertConfig.from_pretrained("indobenchmark/indobert-base-p1", num_labels=3)
        self.model = BertForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", config=config)
        self.model.load_state_dict(torch.load(os.path.join("../model", "model_category.bin"), map_location='cpu'))
        self.results = list()

    def predict(self,text):
        i2w = {0: 'tekno', 1: 'health', 2: 'business'}
        text = text[:512]
        subwords = self.tokenizer.encode(text)
        subwords = torch.LongTensor(subwords).view(1,-1).to(self.model.device)

        logits = self.model(subwords)[0]
        label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

        return i2w[label] , str(f'{F.softmax(logits, dim=-1).squeeze()[label] * 100:.2f}')
    
    def process_text(self,text):
        result = dict()
        label , conf = self.predict(text=text)
        print(f"Success Predict {label}, {conf}")
        # result['text'] = text
        result['label'] = label
        result['conf'] = conf
        self.results.append(result)
        # print(self.results)
    
    def get_result(self):
        return self.results
    def reset(self):
        self.results = list()

class Classifier():
    def __init__(self):
        self.category_classifier = CategoryClassifier()
        self.sentiment_classifier = SentimentClassifier()

    def process(self,text):
        self.category_classifier.process_text(text)
        self.sentiment_classifier.process_text(text)
        return dict(category=self.category_classifier.get_result(), sentiment=self.sentiment_classifier.get_result())
