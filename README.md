# graph-of-words

This repository is PyTorch implementation of Graph Attention Network (GAT) using spectral clustering on text data converted into graph using graph of words concept for classification task.

## Instruction

Keep the label and text tab separated in text file for train and test data. An example of input format is provided below:

```
<label>\t<\text>
Positive\tThis movie is awesome
```

All the configurable parameters along with dataset name are present in `config.py` file.

To prepare the data file and train model, execute the following command:

```sh
$ python main.py
```

## Requirements

This repository relies on Python 3.5 and PyTorch 1.3.0.

## Issues/Pull Requests/Feedbacks

Don't hesitate to contact for any feedback or create issues/pull requests.

For detailed information regarding model architecture, please reach out to me at ghoshsoumava4@gmail.com or refer the literature in reference section.

## References
- [GAT paper](https://arxiv.org/abs/1710.10903)
- [Graph of words](https://frncsrss.github.io/papers/rousseau-dissertation.pdf)
