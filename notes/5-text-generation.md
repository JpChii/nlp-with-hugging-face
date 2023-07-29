# Text-Generation

Transformer language models has an uncanny feature of generating text that is almost indistinguishable from human text. This text generation happens without any explicit supervised leraning, just by predicting the next word based on context in a millions of web pages. With just pretraining LLM's learn a special set of skills and pattern recognition abilites that can be activated with different kind of prompts.

![pretraining-sequence-of-tasks](../notes/images/5-text-generation/pretraining-model-sequence-of-tasks.png)

The image shows addition, unscrambling, translation are some of the sequence tasks that an LLM is exposed during training. This knowledge is transferred during fine-tuning(for larger models during inference-time). These tasks are not chosen specifically ahead of time and occur naturally with huge corpora.

With the advent of GPT-4 and now an open sourced LLAMA2, has given rise to lot's of applications with LLM's at its core with text generation capacity.

In ![5-text-generation.ipynb](../notebooks/5-text-generation.ipynb) notebook we'
ll cover how text generation works with LLM's and how different decoding stratergies impact text generation.

## The Challenge with Generating coherente Text

Until now in the series of notebook, we used a body and a fine-tuned head to get logits. Then we use argmax on logits to get a predicted class or softmax to get prediction probabalites for each token. By contrast, converting the model's probablistic output to text requries a *decoding method*, which introduces a few challenges unique to text generation:

* The decoding is done *iteratively* and requires more compure, not like passing the inputs through forward pass just once.
* The *quality* and *diversity* of text generated depends on the decoding method and associated hyperparameters.

To understand how this decoding process works, let's start by examining how GPT-2 is pretrained and subsequently applied to genreate text.

Like other *autoregressive* or *casual language models* GPT-2 is pretrained to estimate the probabality p(X|Y) of a sequence of tokens **y** = y1, y2,...yt, given some initial context **x** = x1, x2,...xt. Since it's impossible to acquire enough training data, the chain rule of proabality is used to factorize it as a product of *conditional probabalities*.

*Predicting token c given a and b are before it is the conditional probablity intutition*.

![alt contitional-proabablity](../notes/images/5-text-generation/llm-product-of-conditional-probabalities.png)

The note above describe exactly the probablity calculation on right side. This pretraining objective is quite different from BERT's, which utilizer both past and furture contexts to predict a masked token.

We can generate a text by predicting next token, adding it to the sequence and use this as new sequenct to predict next token and continue this iterative process until a special end of sequence token.

Example of this process below,
[text-generation](../notes/images/5-text-generation/text-generation.png)

> **Note:** Since the output sequence is *conditioned* on the choice of input prompt, this type of text genreation is often called as *conditional text generation*.

At the heart of this process lies the decoding method that determines which token is selected at each time step.

A language model produces a logit for each word in  the vocabulary at each time step, we can get the probabality distribution for each token using softmax.

![next-token-softmax](../notes/images/5-text-generation/next-token-softmax.png)

The goal of most decoding methods is to search for the most likely overall sequence by picking a y_hat such that:

![next-token-softmax](../notes/images/5-text-generation/next-token-argmax.png)


Finding y_hat directly involve evaluating every possible sequence with the language model. Since there does not exist an algorithm to do this within an reasonable amount of time we use approximation instead. In this note, we'll explore few of these approximation methods and gradullay build up toward smarter and more complex algorithms that can gernerate high quality texts.

## Greedy Search Decoding

The simplest decoding method to get discrete tokens from a model's continuous output is to greedily select the token with the highest probabality at each timestep.

*Greedy search decoding argmax*
![alt](../notes/images/5-text-generation/greedy-search-decoding.png)