This notes will cover the transformer pipeline use in these notebooks:

# 1. Classification

## Zero-shot-classification

With this pipeline we can leverage a pretrained model and obtain a classification prediction without training.

Get classification prediction for a multi-label task:
```Python
pipe = Pipeline("zero-shot-classification")
output  pipe(sample_text, all_laels_in_use_case, multi_label=True)
# Output returns sequence, labels, scores
```