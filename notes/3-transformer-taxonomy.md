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

Most of the notes will be present in notebooks. Titbits will be added here. Please check the [3-transformer-taxonomy.ipynb](../notebooks/3-transformer-taxonomy.ipynb) for more details

Self-attention mechanism:

In an encoder setup, each token will have information from all other tokens, so the first token [CLS] will have all the information about the entire sequence and can be used as a last hidden state to be fed to find similarity or to decoder or to perform classification.

In an decoder setup, we restrict the access to future using tril to force the model to predict next tokens. This is also called as auto-regressive setup.