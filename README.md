## Advanced Topics in Computational Semantics - Practical

### Introduction 

This repository is a reimplementation of the paper "Supervised Learning of Universal Sentence Representations from Natural Language Inference Data" by Conneau et al. and is part of the course "Advanced Topics in Computational Semantics" at University of Amsterdam.

## Instructions

1) Open a conda prompt and clone this repository as follows:

`git clone https://github.com/apostolikas/UvA-Advanced-Topics-in-Computational-Semantics.git`

2) Create the environment:

`conda env create -f environment.yml`

3) Activate the environment:

`conda activate atcs`

4) You can view the results for the different encoder types (train,validation,test) by accessing the Jupyter Notebook:

`jupyter notebook atcs_results.ipynb`

## Notes

- Make sure to run the utils.py file first in order to download the tokenizer and prepare the datasets.
- Then you can train the model and evaluate it on SNLI by running trainer.py
- You can change the encoder type along with other hyperparameters by changing the arguments in trainer.py (where you can also see a short description of each argument and the available choices).
- In order to evaluate the model on SentEval, you have to run the senteval.py file.
- You can view the results using Tensorboard for a more interactive experience.


## Acknowledgements 

- The [Deep Learning 1](https://uvadlc.github.io) course provides some useful information about using PyTorch Lightning.
- SentEval was cloned from [here](https://github.com/facebookresearch/SentEval).
- You can find the repository of the original paper [here](https://github.com/ihsgnef/InferSent-1).

## Copyright

This project was developed as part of the course "Advanced Topics in Computational Semantics" at the University of Amsterdam. If you are a student at UvA, please first make sure to consult [this page](https://student.uva.nl/en/topics/plagiarism-and-fraud).
