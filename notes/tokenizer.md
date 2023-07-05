# Tokenizers

Tokenizers convert raw text to inputs that can be used to train a model or get predictions from.

Tokenizer converts a sequence of text to tokens(word, character, sub-word) then convert them to integer ids.

## Tokenizers List

* WordPiece --> BERT tokenizer
* SentencePiece --> XLM-R tokenizer


## Methods

* `convert_ids_to_tokens()` --> To convert tokenizer input_ids to string.

## Attributes

* `vocab_size` --> Vocab size of respective transformer
* `max_model_lenght` --> maximum sequence length of transformer
* `input_name` --> input names required for transformer