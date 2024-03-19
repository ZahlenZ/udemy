# Filter Methods
- rely on characteristics of the data to select features
- do not rely on any model
- less computationaly expensive
- usually give lower prediciton accuracy then wrapper methods
- are well suited for a quick screen and removal of irrelevant features
- may select redundant variables because they do not consider the relationships between features


Chi-squred | fisher score
univariate parametric tests(anoava)
mutual information
variance
    - constant features
    - quasi-constant features


variance
correlation
univariate selection

# Wrapper Methods
- use predictive machine learning model to score and select features
- train a new model on each feature subset
- computationally expensive
-usually provide the best performing feature subset for a given algorithm
- they may not produce the best feature combination for a different algorithm

forward selection 
    - start with no feature and add a feature in each iteration
backward selection
    - removes a feature each iteration
exhaustive search
    - try all possible combinations of features

# Embedded Methods
- perform feature selection as part of the model construction process
- consider the interaction between features and algorithm
- less computationally expensive than wrapper methods, they build a single model and select feautres

lasso 
tree importance

# correlation

- perason correlation
    - 1 is highly correlated
    - -1 is highly anti correlated
- spearman correlation
- kendall correlation

# statistical methods
- chi-squared
- fisher score
- anova
- univariate roc-auc / rmse

# chisquare

used to determine if 2 samples of categorical features were extracted from the same population
compares the distributions of the categories

chi-square goodness of fit
    - determines if a categorical variable follows a hypothesized distribution
    - 
chi-square test of independence

1. create contigency table between the categorical variable and the target
2. find the expected distribution
3. calculate the chi-square statistic
4. obtain the p-value

1. rank the features based on the p-vale of chi-square
    - the higher the chi-square or lower the p-value the more predictive the feature is
2. select the top ranking features
    - cutoff is arbitrary

# use a model

use model to predict with one feature at a time,
compare performance of each feature on its own (using any performance metric)
rank the features
select the top ranking features
