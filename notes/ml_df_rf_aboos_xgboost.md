Steps to building ML model
1. problem formulation
    - convert your business problem into a statistical problem
    - clearly define the dependent and independent variables
    - identify whether you want to predict or infer
2. data tidying
    - tranform collected data into a useable data table format
3. pre-processing
    - filter data
    - aggregate values
    - missing value treatment
    - othlier treatment
    - variable transformation
    - variable reduction
4. train-test split
    - training is the information used to train an algorith
    - testing data includes only input data, not the corresponding expcted output.
    - testing data is used to assess the accuracy of model
    - usually 70-80% of teh available data is used as training data
5. model building
6. validation and model accuracy
    - insample error: error resulted from applying your prediction algorith to the dataset you built it with
    - out of sample error: error resulted from applying your prediction algorith to a new data set
7. prediction
    - set pipeline to use your model in real life scenario
    - improve by monitoring your model over time
    - try to automate


Decision Trees
1. regression tree: for continuous quantitative target variable. eg. predicting raingfall, predicting revenue, predicting marks etc.
2. classification tree: for discrete categorical target variables. eg predicting high or low, win or loss..


Basics of regression tree

1. divide the predictor space. that is, the set of possible values for X1, X2..Xp - into J distinct and non-overlapping regions R1, R2,..Rp
2. for every observation that falls into the region Rj, we make the same prediction, which is simply the mean of the reponse values for the training observations in Rj

top-down greedy approach that is known as recursive binary splitting

top down because it begin at the top of the tree adn tehn successively splits the predictor space
each split is indicated via two new brances further down the tree

it is greedy because at each step of the tree-buyilding process, the best split is made at that particular step, rather than looking ahead and picking a split that will lead toa  better tree in some futer step

1. considers all predictors and all possible cut point values
2. calculates RSS for each possiblity
4. select the one with leas RSS
4. continues till stopping criteria is reached
5. prediction is made, prediction is average of the leaf

stopping criteria
1. minimum observations at interal node
minimum numbers of observations required for further split

2. minimum observations at leaf node
minimum number of observation needed at each node after splitting

3. maximum depth 
maximum layes of tree possible

```python
# method to fill na values with the mean
df.fillna({"column": value}, inplace=True)
```

```python
# method to set dummy variables
# drop first makes sure that we aren't falling to dummy variable trap 
# dummy variable trap has one column with 1 and another column with 0 such that anything in column 2 could be guessed by the value in column 1
df = pd.get_dummies(df, columns=["column1", "column2", "columnsn"], drop_first=True)
```

```python
# will select all columns but the listed one
df.loc[:, df.columns != "column"]
```

```python
plt.figure(figsize=(15,10))
plot_tree(reg_tree, filled=True, rounded=True, feature_names=X.columns)
plt.show()
```

### Classification tree

in classification we can use to judge our model
1. classification error rate
2. gini index
    - if 0 there is node purity, the node is only one type of classification
3. gross entropy
    - small values when the node is pure

### Ensemble Methods

1. bagging
    - concept: take many training sets, build a seperate model from each training set, average the resulting predictions to get the final prediction
    - practically we don't have access to this many training sets, so we use bootstrapping to create them
    - while bagging pruning is not done, full length trees are grown
    - indicidual trees have high variance and low bias, averaging reduces the variance
    - in regression we take the average of predicted values
    in classification we take majority vote i.e. most predicted class will be taken as the final prediction
2. random forest
    - problem: bagging creates correlated trees
    - all of the bagged trees will probably use the strongest predictor as the first split so they will all be "roughly the same"
    - concept: we use subset of predictor variables so that we get different splits in each model
    - different set of m predictors out of p where p is all of the predictors. so if m=p then we are just doing bagging
    - Rule of thumb for choosing m
        - for regression p/3
        - for classification sqrt(p)
        - dont forget to use your business knowlede if the variables are highly correlated try a smaller value of M
3. boosting
    - gradient boost
        - slow learning
        - control length of tree
        - decide how many trees to be made
        - boosting can overfit if number of trees is to large
        1. 1st tree calculate residuals
        2. tree after adjusting for residual
        3. final tree after adjusting residuals multiple times
    - ada boost
        1. create a tree assign more weight to misclassified observations
        2. retrain the tree after account for weight
        3. final tree after accounting for weigh N-1 times
    - xgboost
        - almost simliar to gradient boosting
        - xg-boost used a more regularized model formalization to control overfitting
        - for model, it might be more suitable to be called as regularized gradient boosting
        - regularization: the cost function we are trying to optimize also contains a penalty term for number of variables. in a way, we want to minimize the number of variables in final model along with MSE or accuracy. this helps in avoiding overfitting
        - xgboost contain regularization terms in the cost function

problem with normal decision tree
    - high variance


### gathering relevant data
1. identify data need
2. plan data request
3. quality check on data

acquire/build data dictionary
1. definition of predictors
2. unique identified of each table (primary keys)
3. foreign keys or matching keys between tables
4. explanation of values in case of categorical variables

univariate analysis
1. Central tendency
    1. mean
    2. mode
    3. median
2. Dispersion
    1. range
    2. variance
    3. max/min
    4. quartiles
    5. standard deviation
3. count/null count


outlier treatment
Reasons
    - data entry errors
    - measurement error
    - sampling error etc
Impact
    - it increases the error variance and reduces the power of statistical tests
Solution
    - detect outliers using EDD and visualization methods such as scatter plot, boxplots, histograms
    - impute outliers

1. capping and flooring
    - impute all values above 3*p99 and below .3*p1
    - impute with values 3xP99 and .3P1
    - can use a different multiplier instead of 3, as per your business requirement
2. exponential smoothing
    - extrapolate curve between P95 and P99 and cap all the values falling outside to the value generated by the curve
    - similarly, extrapolate curve between P5 and P1
3. sigma approach
    - identify outlier by capturing all the values falling outside mean +/- (x)(stand deviation)
    - you can use any multiplier as x, as per your business requirement

Impact
    - handling missing data is important as many machine learning algorithms do not support dat with missing values.
Solution
    - remove rows with missing data from your dataset
    - impute missing values with mean/median values
Note
    - use business knowledge to take separate approach for each variable
    - it is advisable to impute instead of remove in case of small sample size or large proportion of observations with missing values


1. impute with zero
    - all missing values to 0
2. impute with median/mean/mode
    - for numerical variables, impute missing values with mean or median
    - for categorical variables, impute missing values with mode (category with max frequency)
3. segment based imputation
    - identify felevant segments
    - calculate mean/median/mode
    - impute the missing value according to the segments
    - for example, we can say rainfall hardly varies for cities in a particular state
    - in this case we can impute missing rainfall value of a city with the average of that state

### seasonality
presence of variations that occure at specific regular intervals less than a year, such as weekly, monthly, or quarterly

reasons
    - weather
    - vacations
    - holidays
solution
    = calculate multiplication factor for each month
        m_month = mean_year / mean_month
    - multiply each observation with its multiplication factor


### bivariate analysis
Bivariate analysis is the simultaneous analysis of two variables(attributes). it explores the concept of relationship between two variables, whether there exists an association and the strength of this association, or whether there are differences between two variables and the significance of these differences.

Scatter plot
    - scatter indicates the type (linear or non-linear) and strength of the relationsihp between two variables
    - we will use scatter plot to transform variables
Correlation
    - linear correlation quantifies the strenght of a linear relationship between two numerical variables
    - when there is no correlation between two variables, there is no tendency for the values of one quantity to increase or decrease with the values of the secon dquantity
    - correlation is used to drop non usable variables

identify
    - use business knowledge and bivariate analysis to modify variable
methods
    - use mean/median of variables conveying similar type of information
    - create ratio variable which are more relecant to business
    - transform variable by taking log, eponential, roots etc...


non usable variables
1. variables with a single unique value
2. variables with low fill rate
3. variables with regulatory issue
4. variable with no business sense

### correlation 
is a statistical measure that indicates the extent to which two or mnore variables fluctuate togethr. A positive correlation indicates the extent to which those variables increae or decrease in parallel, negat indicate one variable increase as the other decrease. positive they increase together.

the correlation matrix
definition
    - correlation matrix is a table showing correlation coefficients between variables.
    - each cell in the table shows the correlation between two variables
    - a correlation matrix is used as a way to summarize data, as an input into a more advanced analysis, and as a diagnostic for advanced analysis

look for values > .8 and values < -.8