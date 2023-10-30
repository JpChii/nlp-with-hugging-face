# Transformer Taxonomy

In this respective notebook, below topic'll be covered,
1. Encoder
2. Decoder
3. Encoder-only
4. Decoder-only
5. Encocer-decoder
6. self attention
7. Scaled dot-product attention
8. Mult-headed Attention
9. Layer Normalization
10. Positional embeddings'll be covered
11. Visualizing attentions with BertViz

Most of the notes will be present in notebooks. Titbits will be added here. Please check the [3-transformer-taxonomy.ipynb](../notebooks/3-transformer-taxonomy.ipynb) for indepth details with code.

## Transformer architecture

* Encoder-decoder combines to form transformer.
* Encoder process input through embeddings, multi-head attention, feed forward layer plus normalization to maintain zero mean, uni std between attention and ff layers. 
* The output from ff layer is an hidden state or context representation for each input token. Hence the output size same as input size.

### Encoder

* Embeddings is a lookup table for each token in the entire dataset used. The lookup table size is of the total number of tokens - vocab size. Initially this'll be random and will updated during training.
* Position embeddings is token embeddings augumented with position-dependatn pattern of values arranged in a vector.
* Embeddings and position embeddings are combined together and sent to multi-headed attention to obtain more information on the input text.
* How does this happen? Self-Attention:
    * In self-attention, each embedding will be projected to linear representation of key, query and value.
    * To calculate attention weight for token 0, the query value of this token will be dot product with keys of all other tokens in the sequence.
    * Now this has some information about all the other tokens, these are called attention weights. Attention weights is also a learnable parameter.
    * This attention weight is combined with value of token 0 to get a new contextualized embeddings - context? because now we've some  context about other tokens. During training the context are learned.
    * This is why we use token 0 [CLS] for classification task using transformer encoders, because this token has context from all other tokens in the sequence.
* The entire process above in above point is scaled dot product attention implementation of self-attention.
* Matrix multiplication of query and key is also scaled using embedding to avoid explosion.
* These linear tranformations for key, query and value vectors has it's own set of learnable parameters allowst he model to lean different semantic aspect of the sequence.
* This is a single attention head.
* By having multiple heads the model can learn multiple aspects. This is called multi attention head.
* The head_dim for multi-head attention is calculated by embed_dim / number_of_heads. This is done to maintain the size of embedding dim after passing the embeddings through all heads and concatenating them together.
* And finally comes the feed forward layer. This is a fully connected layer processing each vectors independently insteas of processing them as a sequence.
* For this reason this is also called position-wise-feed-forward layer.
* This is where the capacity and memorization is hypothesized to happen and scaling happens as well.
* After all this we add layer normalization between mha(multi attention heads) and ff(feed forward layer) to maintain zero mean and unit variance.
* There are two versions to this pre and post layer normalizatoin based on whether it comes before or after mha and ff.
* At the end we can add a classification head if we want to use the encoder for classification tasks.ß

### Decoder

* Decoder has the exact same steps as encoder with below exceptions:
    * In decoder the future tokens are not interacted with in attention to avoid lekage with next token prediction
    * This has it's own mha plus eda(encoder-decoder attention). The eha uses key from encoder, query from decoder to calculate the attention weights. The value is from decoder.