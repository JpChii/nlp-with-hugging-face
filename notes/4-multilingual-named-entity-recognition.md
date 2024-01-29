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

> Note: *Zero-short transfer or zero-shot learning* usually refers to the task of training a model on one set of labels and then evaluating it on a different set of labels. In the context of transformers, zero-shot learning may also refer to situations where a lnaguage model like GPT-3 is evaluated on a downstream task that it wasn't even fine tuned on.

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

The architecture and training of these transformers are same a monolingual model, the only exception is a corpus with multilingual data. A model trained on such data even though not informed to differntiate languages, the linguistic representation learned from training performs well across multiple languages for a variety of downstream tasks. Sometimes these model perform better than monolingual models, elimintaing the need to train them.

`Benchamrk Dataset` -> To measure the performance of cross-lingual NER, the below datasets are often used:
    * [CoNLL-2002](https://huggingface.co/datasets/conll2002) 
    * [CoNLL-2003](https://huggingface.co/datasets/conll2003)

### Evalutaion of MultiLingual Transformers

These models can be evaluated in three different ways:

1. `en` -> Fine Tune on English training data and then evaluate on each language set
2. `each` -> Fine Tune on each language and evaluate on monoligual test data to measure per-language performance
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

### Tokenizing a dataset

A NER dataset will typically have tokens(seperated by whitespace words) with respective entity labels using an NER scheme.

* We should use the tokenizer used by the pretrained model
* After tokenizing the words will be split and the labels needs to be aligned with tokenized words
* For this we can leverage a transformers tokenizer called word_ids:
    > This method provides word id for each word in a sequence. When tokenizer split into sub-words, all the sub-words of single word will have same word id which is index of that word when it was split based on space during pretokenization step
* For alignment we can use the follwing recipe:
    * word_id -> None: These are special tokens and we can assign -100 to these
    * word_id of current token == previous token: Then this is a split word, we can assign label_id of I-entity tag or -100 itself
    * If not we'll get the label based on word index and assign it
* We'll set the tag IGN to -100 and use it in post processing to combine the split subwords

Sample function to do the above
```Python
# Let's wrap the above logic into a function to map it to entire dataset
from typing import Dict
def tokenize_and_align_labels(examples)->Dict:
    """_summary_

    Args:
        examples (dataset): dataset input from in hugging face datasets format
    """
    tokenized_inputs = xlmr_tokenizer(
        examples["tokens"], # Split tokens
        is_split_into_words=True, # To not split again
        truncation=True
        )
    labels = []

    # Looping through ner tags list
    for idx, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=idx) # Fetch word ids for the specific index
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


```

> We've to use a id2tag or tag2id functions from tokenizer or implement our own to convert ner_tags from text to ids and vice versa.

## Transformers for Named Entity Recognition

In text classification we pass [CLS] token to dense layers to classify a sequence. In NER all tokens needs to calssified, hence we pass all hidden states to dense layers. Hence NER is also called token classification task.

*Sequence classification*

![alt](https://github.com/JpChii/nlp-with-hugging-face/blob/main/notes/images/4-multilingual-named-entity-recognition/sequence-classification.png?raw=1)

For NER we do the same but with all tokens and assign a label for each token on which entity it is, hence NER is also called as *token classification task*.

*token classification*

![alt](https://github.com/JpChii/nlp-with-hugging-face/blob/main/notes/images/4-multilingual-named-entity-recognition/token-classification.png?raw=1)

## The Anatomy of the Transformers Model Class

Transformers is organized around dedicated classes for each architecture and task. The model associated with different tasks are named according to a `<ModelName>For<Task>` convention ot `AutoModelFor<Task>` when using `AutoModel` classess.

Let's assume we want try out a poc using a model, what if the task is not available with that model. No need to worry that is where Body and Head concept of transformers pitches in...

### Bodies and Heads

Like we saw in [2.text-classification.ipynb](../notebooks/2-text-classification.ipynb) we used DistillBERT's body and trained a classification head.

Here Body is task agnostic as it's set of pretrained weights on a corpus and Head can be attached to the body to leverage the features it has learned and use it to perform our downstream task.

> This is the main concept that makes transformers so versatile. The split of architecture into a `body` and `head`.

This strucuture is reflected in the Transformers code as well: The body of a model is implemented in a class such as `BERTModel` or `GPT2Model` that returns the hidden states of the last layer. Task-specific models such as `BertForMaskedLM` or `BertForSequenceClassification` use the base model and add the necessary head on top of the hidden states.

*Body-Head architecture*
![alt](https://github.com/JpChii/nlp-with-hugging-face/blob/main/notes/images/4-multilingual-named-entity-recognition/body-head.png?raw=1)

This seperation of bodies and heads allow us to build a custom head for any task and just mount it on top of a pretrained model.

To create a custom model, we can use the following recipe:
* Inherit the pretrained model or architecture used by body to use it's pretrained weights
* Load the body's config
* Use the config to do the below in head __init__
    * Set num_labels form config
    * Initialize the body
    * Initialize the dropout with config hidden dropout
    * Initialize classifer with hidden_size from config as input and num_labels as output
    * Then do `init_weights()` to load weights from pretrained model
    * Then implement the forward pass

Sample
```Python
import torch.nn as nn
from transformers import XLMRobertaConfig
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

class XLMRobertaTokenClassification(RobertaPreTrainedModel):
    config_class = XLMRobertaConfig

    def __init__(self, config):
        super().__init__(config)
        # Number of NER tags
        self.num_labels = config.num_labels
        # Load model body
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # Set up token classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        # Load and initialize weights
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None,**kwargs):
        # Use model body to get encoder representations
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)

        # Apply classifier to outputs
        sequence_output = outputs[0]
        logits = self.classifier(self.dropout(sequence_output))

        # Calculate losses
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```

### Loading a custom model

With tagtoid and idtotag functionalities we can load our custom model. `tags` variable holds this information.

Set these to Config using from_pretrained like below:
```Python
from transformers import AutoConfig
xlmr_config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=xlmr_model_name,
    num_labels=tags.num_classes,
    id2label=index2tag,
    label2id=tag2idx,
)
```


Now we've encoded data, custom model. How do we evaluate the model after trainings. Need some Performance Measures to do this:

## Performance Measures

Evaluating NER is similar to classification but all the entity prediction for each token has to be predicted correctly. This can be done with seqeval and this'll return a report for each entity with recall, f1, precision.

Similar to aligning tokens with entities before passing to to model, predictions has to be aligned with labels before evaluating it with seqeval.

Assuming we've batches of predictions, we'll do the below:
* Loop through each sequence
* Inside each sequence:
    * If id is -100 ignore predictions
    * If not used index2tag and pass the id using batch_id and seq_id to get the labels
    * Append truth, pred to their respective lists
* Return the lists after processing all sequences

Sample function:
```Python
def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    labels_list, preds_list = [], []

    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx in range(seq_len):
            # Ignore label IDs = -100
            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
                example_preds.append(index2tag[preds[batch_idx][seq_idx]])

        labels_list.append(example_labels)
        preds_list.append(example_preds)

    return preds_list, labels_list
```

## Fine-Tuning NER

To finetune NER with huggingface Transformers, we need the below things:

1. TrainingArguments
    * To set hyperparameters like epochs, batch_size, evaluation_strategy, save_steps etc.
2. compute_metrics
    * A compute_metrics function to pass to Trainer to calculate metrics.
    * This function will algin the predictions and calculate f1 using seqeval.
    ```Python
    from seqeval.metrics import f1_score
    def compute_metrics(eval_pred):
        y_pred, y_true = align_predictions(eval_pred.predictions, eval_pred.label_ids)
        return {"f1_score": f1_score(y_pred, y_true)}
    ```
3. data_collator
    * Models expect input to be of same length, we've sequences of different lengths. 
    * To make all the sequences of equal length, we'll pad each sequence to the length of longest sequence.
    ```Python
    # This creates a data collator and requires tokenizer
    from transformers import DataCollatorForTokenClassification
    data_collator = DataCollatorForTokenClassification(xlmr_tokenizer)
    ```
4. model_init
    * Function to create an untrained model everytime for fine-tuning.
5. train_data, eval_data
6. tokenizer

*We'll combine all these six components into transformers Trainer*

sample Trainer
```Python
from transformers import Trainer

trainer = Trainer(
    model_init=model_init, # Creates an untrained model
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    train_dataset=panx_de_encoded["train"],
    eval_dataset=panx_de_encoded["validation"],
    tokenizer=xlmr_tokenizer,
)
```

### Error Analysis

Let's say the performance metrics are good after fine-tuning, while in practice it might've some serious flaws. Some examples might be:
* Mask too many tokens and mask labels as well to get a promising loss.
* `compute_metrics()` might've bug that overstimates true performance.
* Including `O` as normal calss which will heavily skew accuracy and F1-score, since it's a majority class by large amrgin.

When model performs worse, performing an error analaysis might give better insight than digging through the code.

For analysis we can look at validation samples with highest loss:
* Pass the inputs to fine-tuned model and get logits
* Get labels
* Calculate cross_entropy with output logits and labels
* return loss and predicted_label
* Convert predicted_label_ids and truth_label_ids to label_names
* Next we'll explode the dataframe to get tokenwise data
    * We can calculate the tokens with highest loss and make some improvements
    * Confusion matrix with truth and preds to get more inference
* We can calculate total sequence loss

After this we can fine-tune and evaluate models based on [above three methods](../notes/4-multilingual-named-entity-recognition.md#evalutaion-of-multilingual-transformers).