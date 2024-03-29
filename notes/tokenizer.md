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
4. Lowercasing
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

### Tokenizer algorithms

* The model splits the words from pretokenizer into sub words.
* Reduce the size of the vocabulary and number of out-of-vocabulary tokens.
* Serveral subword tokenization algorithms exist
    - BPE
        * *BPE Vocabulary creation:*
                1. Starts with basic units(single characters)
                2. Merges most frequently occuring basic units to create new tokens
                3. Adds them to vocabulary
            This process is reiterated until defined vocabulary size is reached
    - Unigram
        * *Unigram Vocabulary creation*:
                1. Starts with all words and potential subwords from corpus
                2. The removes or splits the less useful tokens to reduce vocabulary size
            Repeats step 2 until target vocabulary search is reached.
    - WordPiece[BERT]
        Uses [CLS] and [SEP] tokens to indicate start and end of text
    - Sentence-Piece[XLMRoBERTa]
        Uses <s> and </s> tokens to indicate start and end of text
* Now we've a list of integers(input IDs)

Now text becomes like below,
`["jack", "spa", "rrow"", loves", "new", "york","!"]`

### Postprocessing

* This is the last piece of tokenization pipeline where special tokens are added.
* Like CLS, SEP by BERT tokenizer and <s>, </s> by SentencePiece tokenizer.

## Tokenizers List

* BERT tokenizer --> WordPiece
* XLM-R tokenizer --> SentencePiece
* RoBERTa -> BPE
* QA tokenizers
    1. deepset tokenizers from huggingface are all qa tokenizers
    2. QA tokenizers requires two inputs question, context
    3. Question and context are differntiated after tokenization using token_type_ids, 0 for question tokens, 1 for answer tokens or vice versa


## Methods

* `convert_ids_to_tokens()` --> To convert tokenizer input_ids to string.
* `encode()` --> To tokenize a sequence and convert it to ID's from vocab. This gives tokens as a list.
```Python
# Encoding sequence to input ids
from transformers import XXXTokenier
tokenizer = XXXTokenizer.from_pretrained("checkpoint")
input_ids = tokenizer.encode(text, return_tensors="pt")
```
* `decode()` --> To dekoenizer any tokenized input like input_ids, etc. This gives output as a single string of tokens joined together.
```Python
tokenizer.decode(input_ids[0])
print(tokenizer.decode(inputs["input_ids"][0]))
```

## Attributes/Parameters

* `vocab_size` --> Vocab size of respective transformer
* `max_model_lenght` --> maximum sequence length of transformer
* `input_name` --> input names required for transformer
* `is_split_into_words`(bool) --> If the sequence is already split into tokens, set this to True to convert it to integers without splitting.
* `tokens()` --> Gives back tokens, sequence split to words
```Python
tokens = tokenizer(text).tokens()
```
* `word_ids()` --> This method provides word id for each word in a sequence. When tokenizer split into sub-words, all the sub-words of single word will have same word id which is index of that word when it was split based on space during pretokenization step.
* return_overflowing_tokens -> To enable slidiing window to avoid  trunctaion

## Outputs

* `token_type_ids` --> this is used to differentiate two segments of inputs where it's passed as pair. Ex in QA inputs question tokens will have token_type_ids as 0 and context will have 1.

## Other Tokenizers

### NLTK Sentence Tokenizer

Splits sequences based on sentences

```Python
import nltk
from nltk.tokenize import sent_tokenize
```

This tokenizer is capabale of differentiation difference between a . in abbreviations vs a . at the end of sentence.