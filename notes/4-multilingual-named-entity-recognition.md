# Multilingual Named Entity Recognition

In the series of notebooks till now, we've applied transformers to solve NLP problems on engligh corpora. What if the dataset has multiple languages, mainitainig multiple monolingual models in production will not be any fun.

Fortunatley, we've a class of multilingual transformer to the rescue. Like BERT they are pretrained for masked language modeling as a pretraining objective.

By pretraining on a huge multilingual corpora, we can achieve zero-shot cross lingual transfer. By fine tuning the pretrained label on one language, we can evaluate it on another language without fine tuning for that language explictly.

In this notebook we'll use XLM-RoBERTa pretrained on [2.5Terabyte of text based on Common Crawl Corpus](https://commoncrawl.org/).
The dataset contains only data without parallel texts(translations) and trained an encoder with MLM on this dataset.

Some applications of NER:
* insights from documents
* augmenting quality of search engines
* building a strucutred database from corpus

> Note: *Zero-short transfer or zero-shot learning* usually refers to the task of training a model on one set of labels and then evaluating it on a different set of labels. In the context of transformers, zero-shot learning may also refer to situations where a lnaguage model like GPT-3 is evaluated ona downstream task that it wasn't even fine tuned on.

Problem(assumption):

We want to perfoerm NER for a customer based in Switzerlang, where there are four national languages(With English serving as bridge between them.)

We'll be continuing further notes, code explainations, learning on [4-multilingual-named-entity-recognition notbeook
](../notebooks/4-multilingual-named-entity-recognition.ipynb)