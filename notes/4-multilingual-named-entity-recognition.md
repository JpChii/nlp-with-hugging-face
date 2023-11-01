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

We want to perform NER for a customer based in Switzerland, where there are four national languages(With English serving as bridge between them.)
2003199
We'll be continuing further notes, code explainations, learning on [4-multilingual-named-entity-recognition notbeook
](../notebooks/4-multilingual-named-entity-recognition.ipynb)

## Dataset used
Cross-lingual Transfer Evaluation of Multilingual Encoder (XTREME) called [WikiANN or PAN-x](https://huggingface.co/datasets/wikiann). This dataset has wikipedia articles in multiple languages. We'll use a subset of this dataset which has annotations for ner.

## Annotations

This dataset has three entities -> PER, ORG, LOC annotated in [IOB2 scheme](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)).

## Steps to understand the data

1. Understad the scheme used for annotating entites
2. Calculate the percentage of entites available for each entity, as we've to evaluate the model performance on each entity and not on the whole.

## Multilingual Transformers

The architecture and training of these transformers are same a monolingual model, the only exception is a corpus with multilingual data. A model trained on such data evn though not informed to differntiate languages, the linguistic representation learned from training performs well across multiple languages for a variety of downstream tasks. Sometimes these model perform better than monolingual models, elimintaing the need to train them.

`Benchamrk Dataset` -> To measure the performance of cross-lingual NER, the below datasets are often used:
    * [CoNLL-2002](https://huggingface.co/datasets/conll2002) 
    * [CoNLL-2003](https://huggingface.co/datasets/conll2003)

### Evalutaion of MultiLingual Transformers

These models can be evaluated in three different ways:

1. `en` -> Fine Tune on English training data and then evaluate on each language set
2. `each` -> Fine Tune and evaluate on monoligual test data to measure per-language performance
3. `all` -> Fine Tune on all languages data and evaluate each language

## Models

mBERT is one of the first multilingual transformers but differs from BERT only in corpus(multilinugal wikipedia articles).

XLM-RoBERTa(or XLM-R for short) has supersed mBERT long since:
* MLM of 100 languages is the only pretraining objective
* Corpus is much larger than mBERT with wikipedia dumps for each languages and 2.5TB common web crawl data. This provides good boost to langauges with low resources(data).
* Differs from monolingual RoBERTa with below:
    * Removes next sentence prediction
    * Moves away from language embeddings used in XLM to sentence-tokenizer
    * 25,000 vocab from 55,000 vocab

## Tokenizer

[Tokenizer indepth](../notes/tokenizer.md#tokenizer-pipeline)

### SentencePiece Tokenizer

* Is based on unigram subword segmentation, this encodes each input text as a sequence of Unicode characters
* This unicode character gives sentencepiece the ability to be agnostic about accents, punctuations which useful for multilingual corpus like japaneese with no whitespace characters
* Another feature is usage U+2581 or __ character which allows detokenization without whitespace and without relying on language-specific pretokenizers. Ex New York! wordpiece loses information that there is no space between York and !.