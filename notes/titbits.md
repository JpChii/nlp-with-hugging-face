# Titbits

This contains tips from all the notebooks in this repository.

## 9-dealing-with-few-to-no-labels.ipynb
1. To run a model runs on GPU with transformers pipeline is to use device=0
2. An important aspect for zero-shot classification is the domain we're operating in.
3. Data Augmentation libraries -> [Nplaug](https://github.com/makcedward/nlpaug) and [TextAttack](https://github.com/QData/TextAttack) provides various recipes for token perturbations.
4. FAISS library for efficient embedding search and clustering of dense vectors.

## 4-multilingual-named-entity-recognition.ipynb
1. For mner, the dataset must be created representative of the real word percentage of languages used in a counrty. Ex: In switzerland german, french, italian, english are used 62, 22, 8, 6 percent respectivley. Hence if we create a dataset for mner, the dataset must have the same percentage of data in the dataset created to develop a model.