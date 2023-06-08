# Text Classifiation

Text classification is one of the most common tasks in NLP. It can be used for a broad range of applications, like tagging customer feedback, classifying a ticket to particular queue etc.

## Problem:

Sentiment analysis is another common text classification task. We can build a system to identify emotions based on text.

We'll tackle this with a DistillBERT. The code will be covered in [2-text-classification.ipynb](../notebooks/2-text-classification.ipynb)

*checkpoint* for models corresponds to a set of weights that are loaded into a given transformer architecture.

A typical model fine-tuning with datasets, tokenizers, transformers looks like below,
![alt fine-tuning](images/2-text-classification/fine-tune.png)

## Reading List

1. [Datasets documentation](https://huggingface.co/docs/datasets/index)
2. [Andrei Karpathy's recipe to train neural networks](https://karpathy.github.io/2019/04/25/recipe/)