# Mechanism_of_Action

## Introduction
Scientists seek to identify a protein target which is associated with that disease, to build a molecule that can fix that protein target to help cure the disease. In brief, scientists have given this procedure a label referred to as mechanism-of-action or MoA(in short). The term mechanism of actions means the biochemical interactions through which the drug has its biological effect.

We are supposed to develop algorithms and train models to determine the mechanism of action of a new drug based on the gene expression and cell viability information , 
this is our main objective .

## ML FORMULATION OF BUSINESS PROBLEM

Kaggle Competition : https://www.kaggle.com/c/lish-moa/data 
The dataset provided to to us has been split in training and testing dataset , and our task is to develop an algorithm of model that would automatically predict the labels for every test data .Since the given problem is a multi label classification problem , the test data can have one or more labels. 
Mistake or wrong prediction should be as low as possible otherwise it might lead to serious problem.

## Data set column analysis

The Train dataset comprises of 23,814 row , with 876 features for each . Each row represents a sample which is associated with a unique name sig_id . 
In the dataset , we could observe three categorical features cp_type , cp_time and cp_dose . The cp_time and cp_dose has its contribution balanced in the dataset but the cp_type  is just the opposite.The functions of the categorical features :- treatment/control (indicates whether the experiment is a treatment  or control), dosage( the dose level used in the experiment) , timing(time elapsed between adding the drug and when the measurement was taken). This feature is imbalanced and as stated in the competition the  drugs with cp_type is controllable i.e. cp_type is ctrl_vehicle , that drug won’t perform any Mechanism of action
There are 772 gene expression feature and they are represented by ‘g-’. Here , each gene feature gives the expression of one particular gene. There are 100 cell viability feature and they are represented by ‘c-’ .Here , each cell feature expresses the viability of one particular cell line . The original dataset was normalized with Quantile Normalizaton .Quantile normalization is a technique for making two distributions identical in statistical properties.

There are 206 scored target which we need to predict. In addition , we are also provided with 402 non-scored target which we can use to seek relation with the features or the scored target. There are approximately 5000 unique drugs given to us .
In addition , we are provided with a sample_submission and train_drug csv file which can be put to use.

## Performance metric


The metric used for evaluation is the logarithmic loss function.
For every sig_id given in the dataset we have to predict the probability for each and every MoA. Hence , for N sig_id rows , there would be M targets (MoA) .So total there would be N X M predictions and the score is taken by logloss ,

Here ,
N : Number of sig_id (i = 1,2,3,….N)
M : Number of Scored Targets I.e MoA (m = 1,2,3,….M)
y^i,m :is the predicted probability 
y :is the actual probability 
