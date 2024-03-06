accuracy = (true positive + true negative) / ALL observations

Sesitivity is the true positive rate
true positive (true positive + false negative)
used when we are skewed towards false values

specifity or false positive rate
true negative / (true negative + false positive)
used when we are skewed towrads true values

variance is connected to overfitting


bias is connected to underfitting
really high errors
one thing in mind and apply it everywhere

# parameter tuning

number of round: how many times do we want the analysis to be run
ETA: learning rate. how fast do you want the model to learn

minimum child weight: relates to the sum of the weightrs of each observation. low values can mean that maybe not a lot of observations are in the round

max depth: how big should the tree be? bigger trees go into more detail

gamma: how fast should the tree be split?

subsample: share of observations in each tree?

colsample by tree: how much of the tree should b eanalysed per round?

cross validation allows us to find the sweet spot.
Imagine taking the training and test sets, and then further split them, so that you have multiple sets of training sets from the original training set, and multiple sets of test sets from the original training set. use a subset of the training set and validate against the training sets, do this with with all of the sub training sets

# SHAP values

from a business perspective this is really the only thing that matters in the end

bring interpretability to the machine learning model, helps to reduce the idea of it being a "black box"

SHAP values help to show the feature importance

visual impact, very easy to interpret because the feature importance is shows in the visualization

remember correlation is not causality

