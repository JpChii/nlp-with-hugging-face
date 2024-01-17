# Datasets

Hugging Face Datasets, This markdown will serve as a place to recall important points on Datasets library.

*Abbreviation used*
hub --> Hugging face Hub

* `load_dataset()` function can be used to load any dataset from hub
* When dealing with dataset with multiple domains, we can use `get_dataset_config_names()` function to identify the subsets that are available.
* To access a particular subset, `name`  argument with suffix name as value must be passed.
    ```Python
    # Loading PAN-X german subset from xtreme
    from datasets import load_dataset
    load_dataset("xtreme", name="PAN-X.de")
    ```
* In each `Dataset` object, the keys correspond to column names of an Arrow table and values denote the entries in each column.
* To access labels of a dataset feature or column
    ```Python
    # Accessing ner_tags of a subset of xtreme dataset
    tags = panx_ch["de"]["train"].features["ner_tags"].feature
    # For clinc dataset
    clinc["test"].features['intent'].names 
    ```
    Check the ClassLabel column to access the labels
* Dataset offers a `ClassLabel` class for labels, using `ClassLabel.int2str()` we can convert our numerical labels or ner_tags to strings for better visualization of data
    ```Python
    # This code will add ner_tags_str to the dataset and creates a text representation of ner tags
    def create_tag_names(batch):
        return {"ner_tags_str": [tags.int2str(idx) for idx in batch["ner_tags"]]}
    panx_de = panx_ch["de"].map(create_tag_names)
    ```
* To shuffle a dataset to avoid bias, call `shuffle()` on the desired split on  `Dataset`
```Python
dataset = load_dataset("some-dataset")
dataset_shuffled = dataset["split"].shuffle()
```
* To select random number of samples from the split, call `select()` on the desired split on `Dataset`
```Python
dataset = load_dataset("some-dataset")
dataset_selected = dataset["split"].select(subset_of_indexes)
```


## Metric

Datasets library holds Metric class, which provides access to implementations of metrics like seqeval, bleu.

```Python
# Access untraditional metrics
from datasets import load_metric
seq_eval = load_metric("seqeva")
```

## Datasets

This'll cover the datasets used in the series of notebooks in this section.

1. Cross-lingual Transfer Evaluation of Multilingual Encoder (XTREME) benchmark called [WikiANN or PAN-x](https://huggingface.co/datasets/wikiann). This dataset has wikipedia articles in multiple languages.
2. `Benchamrk` -> To measure the performance of cross-lingual NER, the below datasets are often used:
    * [CoNLL-2002](https://huggingface.co/datasets/conll2002) 
    * [CoNLL-2003](https://huggingface.co/datasets/conll2003)