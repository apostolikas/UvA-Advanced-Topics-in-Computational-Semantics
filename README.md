## Advanced Topics in Computational Semantics - Practical 

### Introduction 

This repository is a reimplementation of the paper "Supervised Learning of Universal Sentence Representations from Natural Language Inference Data" by Conneau et al. and is part of the course "Advanced Topics in Computational Semantics" at University of Amsterdam.

## Instructions

1) Open a conda prompt and clone this repository as follows:

`git clone https://github.com/apostolikas/UvA-Advanced-Topics-in-Computational-Semantics.git`

2) Create the environment:

`conda env create -f env.yml`

In case you are trying to run this code on Lisa, create the environment using the follow command:

`conda env create -f environment_lisa.yml`

3) Activate the environment:

`conda activate atcs`

4) For SentEval there's no need to clone the repository as long as I provide the necessary scripts, but you have to download the data. In order to do this, run (in SentEval/data/downstream/):

`./get_transfer_data.bash`

This is going to take a few minutes but once it's done, it's all set.

5) Run the scripts with the following order:
- utils.py (Prerequisities for the SNLI Dataset)
- train.py (Train and evaluation on SNLI)
- eval.py (Evaluation on SentEval)

6) You can view the results train,validation,test) and some error analysis for the different encoder types by accessing the Jupyter Notebook:

`jupyter notebook analysis.ipynb`

## Notes

- Make sure to run the utils.py file first in order to download the tokenizer and prepare the datasets.
- Then you can train the model and evaluate it on SNLI by running train.py.
- You can change the encoder type along with other hyperparameters by changing the arguments in train.py (where you can also see a short description of each argument and the available choices).
- In order to evaluate the model on SentEval, you have to run the eval.py file.
- You can view the results using Tensorboard for a more interactive experience using the tensorboard logs created by the train.py.


## Acknowledgements 

- The original SentEval was cloned from [here](https://github.com/facebookresearch/SentEval).
- You can find the repository of the original paper [here](https://github.com/ihsgnef/InferSent-1).

## Copyright

This project was developed as part of the course "Advanced Topics in Computational Semantics" at the University of Amsterdam. If you are a student at UvA, please first make sure to consult [this page](https://student.uva.nl/en/topics/plagiarism-and-fraud).
