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