# Tokenizers

Tokenizers convert raw text to inputs that can be used to train a model or get predictions from.

Tokenizer converts a sequence of text to tokens(word, character, sub-word) then convert them to integer ids.

## Tokenizer Pipeline

![alt tokenizer pipeline](../notes/images/4-multilingual-named-entity-recognition/tokeinzier-pipeline.png)

Assume the text `Jack Sparrow loves New York!` is passsed through the pipeline.

### Normalization

This step corresponds to the set of operatins applied to make it `cleaner`. Some of them might be as follows,

1. Stripping whitespace
2. Removing accented characters
3. [Unicode normalization](https://unicode.org/reports/tr15/)
    - Is a normalization applied by many tokenizers to deal with the fact that there are many ways to write the same character.
    - This can make two versions of same string appear different.
    - This uses schemes like NFC, NFD, NKFC and NFKD replace the ways to write the same character with standard forms.
4. Lowecasing
    - If the model accpets and uses only lowecase, this will reduce the size of the vocabulary

After normalization the text would look like,
`jack sparrow loves new york!`

### Pretokenization

* Pretokenizatoin gives the upper bound to what the tokens will be at the end of training.
* One way to think about this is splitting a string into words based on whitespace which works well for Indo-European lanuages. Then these words can be split to simpler sub-words with Byte-Pair Encoding or Unigram algorithms in the next step.
* Splitting based on whitespace and grouping them into semantic units is non-deterministic for languages like Chinese, Japanese, Korean.
* Best approch is to pretokenize using a language-specific library.

After pretokenization,
`["jack", "sparrow", "loves", "new", "york","!"]`

### Tokenizer model

* The model splits the words from pretokenizer into sub words.
* Reduce the size of the vocabulary and number of out-of-vocabulary tokens.
* Serveral subword tokenization algorithms exist
    - BPE
    - Unigram
    - WordPiece
* Now we've a list of integers(input IDs)

Now text becomes like below,
`["jack", "spa", "rrow"", loves", "new", "york","!"]`

### Postprocessing

* This is the last piece of tokenization pipeline where special tokens are added.
* Like CLS, SEP by BERT tokenizer and <s>, </s> by SentencePiece tokenizer.

## Tokenizers List

* WordPiece --> BERT tokenizer
* SentencePiece --> XLM-R tokenizer


## Methods

* `convert_ids_to_tokens()` --> To convert tokenizer input_ids to string.

## Attributes

* `vocab_size` --> Vocab size of respective transformer
* `max_model_lenght` --> maximum sequence length of transformer
* `input_name` --> input names required for transformer
* `is_split_into_words`(bool) --> If the sequence is already split into tokens, set this to True to convert it to integers without splitting.