# Summarization

Summarization requires a range of abilites, such as understanding long passages, reasoning about the contens, and producing fluent text that incroprates the main topics from original document. Morever summaraizing a new article is different compared to legal document and each of them requires certain degree of domain generalization. For these reasons summarizing is a difficult task for natural language models, including transformers. Despite these challenges, text summarization has huge applications,

* Helps domain experts to speed up the workflow
* Helps enterprise to summarize contracts, internal knowledge
* Generate content for social media releases and more.

Summarization is a classic sequence-to-sequence(seq2seq) task with an input text and a target text. This is where encoder-decoder transformer excel.

In this notebook we'll cover,

* The challenges involved
* Pretrained transformers to summarize documents, (i.e) our own encoder-decoder model to condense dialouges between several people into a crisp summary.

***Dataset to be used in this notebook*** --> [CNN/DailyMail corpus](https://huggingface.co/datasets/cnn_dailymail)

## Summarization models

#### GPT-2

In [5-text-genreation.ipynb](https://github.com/JpChii/nlp-with-hugging-face/blob/main/notebooks/6_Summarization.ipynb) we checkout how GPT-2 can generate text with some input prompt. Because GPT-2 models are trained on web which includes reddit articles it's good on summarization as well. We can trigger summarization with GPT-2 by appending "TL:DR" at the end of propmt. "TL;DR" is often used to indicate blogs that are too long didn't read to indicate a short version of the long post.

We'll start the summarization experiment by with `pipeline()` from transformers.

#### T5

T5 developers performed a comprehensive study of transfer learning and found they could create a universal transformer by formulating all tasks as text-to-text-tasks.

The T5 checkpoints are pretrained on mixture of unsupervised data(to reconstruct masked words) and supervissed data for several tasks, including summarization. These checkpoints can be directly used for inference using the same prompts used during pretraining. Few input format's as follow,

* To summarize `"Summarize: <ARTICLE>"`
* To translate `translate English to German: <TEXT>`

This capabality makes T5 extremely versatile and with single model we can solve many tasks.

Let's load this and test it out.

*All T5 capabalities*
![alt t5](../notes/images/6-summarization/t5-capabalities.png)

#### BART

BART also uses an encoder-decoder architecture and is trained to reconstruct corrupted inputs. It combines the pretraining schems of bert and GPT-2. We'll use the `facebook/bart-large-cnn` checkpoint, which has been specifically fine-tuned on CNN/DailyMail dataset

#### PEGASUS

The authors argue that the pretraining objective needs to be closer to the downstream task, the more effective it is.
With summarization as the objective instead of general language modelling, they masked the sentence that contain most of the info of their surrounding paragraphs(using summarizaiton evaluation metrics as a heuristic for content overlap) and pretrained PEGASUS model to reconstruct the senteces to obtain sota model for text summarization.

PEGAUS is an encoder-decoder transformer with it's pretraining objective to predict masked sentences in multisentence texts.