- qualitative (categoical)
    - nominal (no order)
    - ordinal (with order)

- Nominal
    - discrete
    - continuous

- statistics
    - descriptive
    - inferential

- supervised learning
    - have inputs and labels
    - ANN
    - CART
    - NAIVE
    - Neural Networks
    - EnsembleBayes
    - Discriminant
    - REgression
    - KNN
    - Forest
    - logistic
    - SVM
    - CHAID

- unsupervised learning
    - only have inputs no labels
    - clustering
        - kmeans
        - hierarchical clustering
    - hidden markov models (HMM)
    - Dimenstion reduction 
        - factor analysis
        - PCA
    - feature extraction methods
    - self_organizing maps
        - nuerual nets


Building Models
1. problem forumation
2. data tidying
3. pre-processing
    - filter data
    - aggregate values
    - missing value treatment
    - outlier treatment
    - variable transformation
    - variable reduction
4. train_test split
    - validation set approach
        - random division of data into two parts
        - usual split is 80:20 (train:test)
        - use when large number of observations
    - leave one out cross validation
        - leaving one observation out every time from training set
    - k-fold cross validation
        - divide data into k parts
        - use k-1 parts for training and 1 part for testing
        - repeat k times
        - average the results
5. model building
6. validation and performance metrics
    - in sample error
        - error resulted from applying your prediction algorithm to the dataset you build it with
    - out of sample error
        - error resulted from applying your prediction algorithm to a new data set
7. prediction
    - setup a pipeline to use your model in real life scenario
    - improve by monitoring your model over time
    - try to automate

repeat steps 5/6 until the model is performing well

Data preprocessing
1. gather business knowledge
2. data finding - where does it come from and acquiring
3. dataset and data dictionary
4. univariate analysis
5. outlier treatment
6. missing value imputation
7. seasonality in data
8. bi-variate analysis and variable transformation
9. non-usable variables
    - variables with single unique value
    - variables with low fill rate
    - variables with regulatory issue
    - variable with no business sense
10. dummy variable creation/handling qualitative data
11. correlation analysis

Bias and variance tradeoff
    - E(epsilon)
        - variance of error, irreducible
    - E(variance)
        - amount by which predicted function will change if we change the training data
    - E(bias)
        -Error due to approximation of comples relationship as a simpler model such as linear model

Goal is to find the point at which the bias and variance are balanced (reduce the sum of the two)

Shrinkage methods
    - ridge regression
        - use a shrinkage penalty which is lamba * sum(coefficients^2)
        - lambda is a tuning parameter
        - lambda = 0 is the same as linear regression
        - because of this shrinkage penalty, ridge regression varies with scale of independent variable, therefore, we need to stadardize the variables
    - lasso regression
        - use the same shrinkage penalty as ridge regression but with the absolute value of the coefficients
        - lasso regression can be used for variable selection
        - lasso regression can be used for feature selection
        - lasso regression can be used for regularization
        - lasso regression can be used for reducing overfitting
        - lasso regression can be used for reducing the complexity of the model
        - in the lasso technique, for sufficiently large value of lambda, several coefficients will actually become zero which reults in variable selection and a simpler model
    - elastic net
    - principal component regression
    - partial least squares

Heteroscedasticity
    - assumption: variance of error term is independent of values of Y
    - solution: scaled down Y variable by using log(y) or sqrt(y)
    - none constant variance is known as heteroscedasiticity

type 1 error 
    - false positive
type 2 error 
    - false negative

There is always a trade off to making these types of errors, is one going to really destroy the analysis or is one not going to be harmful

false pos rate - fp/n - type 1 error, specificity
true pos rate - tp/p - type 2 error, sensitivity, recall, power
pos pred value - tp/p - false discovery proportion
neg pred value - tn/n


### test train split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

### linear regression

```python
import statsmodels.api as sm
X = sm.add_constant(df["independent"])
lm = sm.OLS(df["dependent"], X).fit()

lm.summary()
```

```python
from sklearn.linear_model import LinearRegression
y = df["dependent"]
# sklearn needs to dimension array (n, 2)
X = df[["independent"]]

lm2 = LinearRegression()
lm2.fit(X, y)

print(lm2.intercept_, lm2.coef_)

lm2.predict(X)

sns.jointplot(
    x = df["independent"],
    y = df["dependent"],
    data = df,
    kind = "reg"
)
```

### multiple linear regression

```python
import statsmodels.api as sm
X = sm.add_constant(df[["independent1", "independent2"]])
# can also use
# X_multi = df.drop("dependent", axis=1)
# X_multi_const = sm.add_constant(X_multi)
y_multi = df["dependent"]

lm_multi = sm.OLS(y_multi, X_multi).fit()
lm_multi.summary()
```

```python
lm3 = LinearRegression()
lm3.fit(X_multi, y_multi)
print(lm3.intercept_, lm3.coef_)
```

```python
from sklearn.metrics import r2_score
r2_score(y_multi, lm3.predict(X_multi))
```

### ridge and lasso regression

```python
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklean.model_selection import validation_curve

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

lm_ridge = Ridge(alpha=0.5)
lmridge.fit(X_train_s, y_train)

r2_score(y_test, lmridge.predict(X_test_s))

# setting values of lambda to test
param_range = np.logspace(-2, 8, 100)

# for each value of lambda we are getting three values of R2
# validation curve is running kfold cross validation behind the scene
train_scores, test_scores = validation_curve(
    Ridge(),
    X_train_s,
    y_train,
    "alpha",
    param_range,
    scoring="r2"
)
# take a mean score of the three values of R2
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
# will give us the highest R2 score
max(test_mean)
# only interested in model with highest R2 score
sns.jointplot(
    x=np.log(param_range),
    y=test_mean
)
# find index of test_mean with highest value
n = np.where(test_mean==max(test_mean))
# find lambda value of the index
param_range[n]

lm_r_best = Ridge(alpha=param_range[n])
lm_r_best.fit(X_train_s, y_train)
r2_score(y_test, lm_r_best.predict(X_test_s))
```

```python
lm_lasso = Lasso(alpha=0.4)
# rest of execution is the same as ridge
```

### simple logistic regression

```python
from sklean.linear_model import LogisticRegression
X = df[["independent"]]
y = df["dependent"]

clf_lrs = LogisticRegression()
clf_lrs.fit(X, y)

clf_lrs.coef_
clf_lrs.intercept_
```

```python
import statsmodels.discrete.discrete_model as sm
X = sm.add_constant(df["independent"])
logit = sm.Logit(df["dependent"], X).fit()

logit.summary()
```

### multiple logistic regression

```python
X_const = sm.add_constant(df[["independent1", "independent2"]])
y = df["dependent"]

clf_lr = LogisticRegression()
clf_lr.fit(X_const, y)
clf_lr.coef_
clf_lr.intercept_

# give the probability from the logister curve (sigmoid function)
clf_lr.predict_proba(X_const)
# give the class of the prediction
clf_lr.predict(X_const)

y_pred_03 = (clf_lr.predict_proba(X_const)[:,1] >= 0.3).astype(bool)
```

### scoring

```python
precision_score(y, y_pred)
recall_score(y, y_pred)
roc_auc_score(y, y_pred)
```




