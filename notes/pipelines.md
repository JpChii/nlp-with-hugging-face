This notes will cover the transformer pipeline use in these notebooks:

# 1. Classification

## Zero-shot-classification

With this pipeline we can leverage a pretrained model and obtain a classification prediction without training.

Get classification prediction for a multi-label task:
```Python
pipe = Pipeline("zero-shot-classification")
output  pipe(sample_text, all_labels_in_use_case, multi_label=True)
# Output returns sequence, labels, scores
```

# 2. Summarization

## Prompt Summarization

We can load `text-generation` pipeline with gpt model. Add "\nTL;DR:\n" at the end of text to be summarize to get summary of the context. Since GPT-2 models are trained on web which includes reddit articles. Long Summary TL;DR Short summary. This triggers a summarization with gpt models.

```Python
pipe = Pipeline("text-generation", model"gpt2-large)
gpt2_prompt = sample_text +  "\nTL;DR:\n"
pipe_out = pipe(
    gpt2_prompt,
    max_length=512, # length of summary
    clean_up_tokenization_spaces=True,
)
```

`generated_text` key in pipeline output has the summary output.

## T5 Summarization

We can use T5 model checkpoint to obtain summaries of a sentence. Input format for direct usage:
"Summarize: <Text>"
To use with pipelines:

```Python
pipe = pipeline("summarization", model="t5-large)
out = pipe(sample_text)
```

`summary_text` will've the summarized output.

# 3. Question Answering

We can use any qa checkpoint(from deepset or others) and use this pipeline to get answers.

```Python
from transformers import pipeline
pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
pipe(question=question, context=context, topk=3)
```