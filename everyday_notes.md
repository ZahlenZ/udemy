# Libraries

```python
import matplotlib.gridspec as gricspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

from datetime import datetime
from feature_engine.selection import (
    DropConstantFeatures, 
    DropDuplicateFeatures,
    DropCorrelatedFeatures,
    SmartCorrelatedSelection,
    SelectBySingleFeaturePerformance,
    SelectByTargetMeanPerformance,
    SelectByShuffling,
    RecursiveFeatureElimination
)
from mlxtend.feature_selection import (
    SequentialFeatureSelector as SFS,
    ExhaustiveFeatureSelector as EFS
)
from numpy.random import default_rng
from scipy.stats import chi2_contingency
from sklearn import svm, tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import(
    BaggiingClassifier,
    RandomForestClassifier,
    GradientBoosting,
    AdaBoostClassifier
)
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
    SelectKBest,
    SelectPercentile,
    f_classif,
    f_regression,
    SequentialFeatureSelector as SFS,
    SelectFromModel,
    RFE
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Lasso
)
from sklearn.metrics import (
    accuracy_score,
    confustion_matrix,
    mean_squared_error,
    r2_score
)
from skleanr.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_validate
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preproccessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import plot_tree
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
```


# pandas and numpy

```python
# Importing Data
pd.read_csv() 
# import the data as a series instead of dataframe (time series with only one attribute)
pd.read_csv().squeeze("columns")

# pandas series
pd.series([1, 2, 3], dtype=np,int8, name="numbers")

# EDA
df.head(n)
df.tail(n)
df.sample(n)
df.shape()
df.info(memory_usage="deep")
df.describe()
df.isna().sum()

# Filtering
# Use Loc primarily
df.loc[logical statement]
df.loc[:, [column list]]
df.loc[:, "column1":"columnn"]
df.iloc[[index or index range], :]
df.drop([colum list], axis=1)
df.drop_duplicates(
    subset=[list of columns],
    keep=["last", "first"],
    ignore_index=True
)
avg_sales = 45
df.query("family in [column list] and sales > @avg_sales)

# Manipulation
pd.get_dummies(data, columns=[column list])
df["column"].fillna(with what)
np.array()
np.newaxis
df.astype(
    {
        "colun": "datatype",
        "columnn": "datatype"
    }
)
df.rename(
    columns = {
        "oldname": "newname
    },
    inplace=False
)
mapping_dict = {
    "oldvalue": "newvalue"
}
df["column"] = df["column"].map(mapping_dict)
df.assign(
    column_name = calculation,
    column_name = calculation
)
df.groupby([column list], as_index=False)["column"].sum()
(
    df
    .groupby([column list])
    .agg(
        sales_sum=("column", "sum"),
        sales_avg=("column", "mean")
    )
)
# id_vars is the column that will be repeated down the new dataframe
# value_vars is the columns that will be melted
# each unique id_var has a value for the value_vars 
# table with have unique combinations of id_var and value_var
df.melt(
    id_vars="column",
    value_vars=[list],
    var_name="name",
    value_name="name"
)

# math
df.corr()
np.percentile(series, [percentile list as 75, 99...])
np.log(series)
np.median(array)
np.unique(array)

# aggregation
.count()
.first()
.last()
.mean()
.min()
.max()
.argmax() # index for the smallest or largest
.argmin()
.argsort(ascending=True) # index array that would sort ascending
.std()
.var()
.prod()
.sum()
.quantile()
.unique()
.nunique()
.value_counts() # unique items and frequency

# membership
df.index.isin(list)
df.loc[df["column"].isin(list)]

# random numbers
rng = default_rng(seed=289)
rng.random(n)
rng.normal(mean, stdev, n)
rng.integers(low, high, n)

# array methods
array.dtype
array.round(n)
array.sum()
array.mean()
array.std()
array.reshape(row, col)
array.sort()
array.sort_values(ascending=True)
array.sort_index(ascending=True)

# indexing
df.index
df.index = [list]
df.reset_index(drop=True)
df.reindex(labels = ["new", "column", "order"], axis=1)

# pivot tables
df.pivot_table(
    index="column", # category on left of pivot
    columns="column" # categories of the top pivot
    values="column", # column aggregated values come from
    aggfunc=("sum", "mean", "max"), # aggregations to use
    margins=False # show row and column total when true
    # none highest values in all pivot table
    # 1 highest values in each row
    # 0 highest values in each column
    axis=["none", 1, 0] 
)

df.pivot_table().style.backgrou_dradient(cmap="", axis=None)

# joining
left_df.merge(
    right_df,
    how="",
    left_on=[columns],
    right_on=[columns]
)
```


## seaborn and matplotlib

### matplotlib overview
```python
fig, ax = plt.subplots(
    figsize=(width, height) # default 6.4 x 4.8 (in inches)
)
# plot it
ax.plot(
    df.index, 
    df["series1"],
    label="what to call it",
    color="green",
    linewidth=2,
    linestyle="--"
)
ax.plot(
    df.index, 
    df["series2"],
    legend="what to call it",
    color="red",
    linestyle=":"
)
# titles
fig.suptitle(
    "Overall Title",
    fontsize=16
)
ax.set_title(
    "Chart Title",
    fontsize=10
)
# set axis labels
ax.set_xlabel(
    "date",
    fontsize=10
)
ax.set_ylabel(
    "something",
    fontsize=10
)
# set axis lim
ax.set_xlim(lower, upper)
ax.set_ylim(lower, upper)
# custom ticks
ax.set_xticks([iterable]) # example df.index[::2] every other 
plt.xticks(rotation=45) # throw a little of the matplotlib API in here
# Set some other lines
ax.axvline(
    18341,
    c="black".
    ls="--",
    label="Important Date", # adding label to an ax method will auto it into the legend
    ymin=, # adding ymin and ymax will shorten the line
    ymax=,
)
# add some text
ax.text(
    x-coordinate, # if date then epoch days
    y-coordinate,
    string # what you want it to say
)
# add a text annotations
ax.annotate(
    string,
    xy=,
    xytext=,
    arrowprop=dict(
        facecolor=,
        width=,
        headwidth=,
        connectionstyle="angle3, angleA=290, angleB=0"
    ),
    verticalalignment="center"
)
# remove some borders
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
# set legend
ax.legend(
    [list that corresponds to series order],
    loc="lower center",
    # bbox_to_anchor=()also a way to anchor the legend
    ncol=2,
    frameon=True # set to false to remove the frame
) # for custom names in legend

# remember if not in jupyter you need
plt.show()
```

```python
# lollipop charts
fig, ax = plt.subplots()
ax.stem(
    data=df,
    x="", 
    y="",
    basefmt=" ",
    label=["columns in legend"]
)
```

```python
# customized grid
fig = plt.figure(figsize=(width, height))
gs = gridspec.GridSpec(ncols=4, nrows=4)
ax1 = fig.add_subplot(gs[0:4, 0:2]) # slice of rows then columns
```

### customiztion
```python
sns.despine()
```

### figure level seaborn

```python
sns.relplot(
    data=df,
    kind=["scatterplot", "lineplot"],
    hue="column to segregate by"
)

sns.displot(
    data=df,
    x="column"
    kind=["hist", "kde", "edcf", "rug"],
    # seperate by category
    hue="column",
    # use multiple to keep distributions in same figure
    multiple=["stack"],
    # use col to create a facet grid where each column is a different category
    col="column"
)

sns.catplot(
    kind=[
        "stripplot", 
        "swarmplot", 
        "boxplot", 
        "boxplot", 
        "violinplot", 
        "pointplot", 
        "barplot"
    ],
    hue="column to segregate by"
)
```

### Axis level Seaborn

```python
sns.lineplot(
    data=df,
    x="column",
    y="column",
    estimator=sum, # default is mean
    hue="",
    palette="",
    ci=None, # True for confidence intervals
    ls="", # line style
    color=""
)

sns.barplot(
    data=df,
    x="",
    y="",
)

sns.boxplot(
    data=df,
    x="",
    y=""
)

sns.scatterplot(
    data=df,
    x="",
    y="",
)

# scatter with regression plot
sns.regplot(
    data=df,
    x="",
    y=""
)

# extension of regplot
sns.lmplot(
    data=df,
    x="",
    y=""
)

sns.jointplot(
    data=df,
    x="column",
    y="column,
    hue="column"
)

sns.histplot(
    data=df,
    x="columns",
    kde=True
)

sns.kdeplot(
    data=df,
    x="columns"
)

sns.countplot(
    data=df,
    x="column"
)

sns.pairplot(
    data=df,
    diag_kind=["kde"],
    corner=True
)

sns.swarmplot(
    data=df,
    x="",
    y="",
    ax=ax,
    hue="",
    dodge=True
)
```

### Correlation

```python
df.corr(method=["pearson", "spearman", "kendall"])
```

```python
sns.heatmap(
    data=df.corr(),
    # Not sure 
    vmax=number,
    # Not sure 
    square=True,
    cmap="color map",
    # Not sure
    cbar=True,
    # add correlation number to matrix,
    annot=True,
    # format annotation
    fmt=".2f",
    # addition annotation formating
    annot_kws={"size": 10},
    yticklabels=[list of names(typically columns)],
    xticklabels=[list of names]
)
```

```python
# only get the top n correlated variables
corrmat = df.corr()
cols = corrmat.nlargets(n, "label")["label"].index
l_corr = df[cols].corr()
sns.heatmap(
    l_cols,
    cbar=True,
    annot=True,
    square=True,
    fmt=".2f",
    annot_kws={"size": 10},
    yticklables=cols.values,
    xticklables=cols.vaues
)
```

getting the largest k correlated
```python
k = 10
cols = corrmat.nlargest(k, 'dependent')['dependent'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(
    cm, 
    cbar=True, 
    annot=True, 
    square=True, 
    fmt=".2f", 
    annot_kws={"size": 10}, 
    yticklabels=cols.values, 
    xticklabels=cols.values
)
plt.show()
```

### facet grids

```python
g= sns.FacetGrid(
    data=df,
    column="",
    columnwrap=""
)
g.map_dataframe(sns.plottype, x="column")
```

### matplotlib integration

```python
fig, ax = plt.subplots(nrows=n, ncols=m, figsize=(width, height))
sns.plottype(
    ax=ax[n][m]
)
```

### plot

```python
res = stats.probblots(series, plot=plt)
```

# Sklearn


### Processing
```python
scaler = StandarScaler()
scaled_array = scaler.fit_transform(np.array()[:, np.newaxis])

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    Y,
    test_size=0.2,
    random_state=289
)
```

### Models

```python
# Decision tree

# Regression
reg_tree = tree,DecisionTreeRegressor(
    max_depth=3
)


# classifier
clf_tree = tree.DecisionTreeClassifier(
    max_depth=3
)

# random forest
rf_clf = RandomForestClassifier(
    n_estimators=1000,
    n_jobs=-1,
    random_state=289
)

# bagging
bag_clf = BaggingClassifier(
    estimator=model,
    n_estimators=1000,
    bootstrap=True,
    n_jobs=-1, # for all available cores, or n cores you want to use
    random_state=289
)
bag_clf.fit(X, y)

# LDA
clf_lda = LinearDiscriminantAnalysis()
clf_lda.fit(X, y)

# gradient boosting
gbc_clf = GradientBoostingClassifier(
    learning_rate=.02,
    n_esimators=1000,
    max_depth=n
)
gbc_clf.fit(X, y)

# ada boost
ada_clf = AdaBoostClassifier(
    learning_ratte=0.02,
    n_estimators=1000
)
ada_clf.fit(X, y)

# KNN Model
# sensitive to wide distributions need to scale or normalize data before using
clf_knn = KNeighborsClassifier(n_neighbors=1)
clf_knn.fit(X_train, y_train)

# Regression SVM Linear
# default kernal is RBF (radial) documentation has the other methods
svr = SVR(kernel="linear", C=1000)
svr.fit(X_train, y_train)

# Classification Model
clf_svm = svm.SVC(kernel="linear", C=0.01)

# Classification model
clf_svm = svm.SVC(kernel="polynomial", degree=2, C=0.01)

# Classification model
clf_svm = svm.SVC(kerneal="rbf", gamma=0.5, C=10)

# grid search
params={
    "param1": [values],
    "param2": [values]
}
grid_search=GridSearchCV(
    estimator=model,
    param_grid=params,
    n_jobs=-1,
    cv=5 # cross validations
    scoring=["accuracy"]
)
grid_search.fit(X, y)
grid_search.best_params_
grid_search.best_estimator_

# evaluating
model.fit(X_train, y_train)
y_train_pred = reg_tree.predict(np.array())
mean_squared_error(y, y_pred)
r2_score(y, y_pred)
confustion_matrix(y, y_pred)
accuracy_score(y, y_pred)

plt.figure(figsize=(width, height))
plot_tre(model, filled=True, rounded==True, feature_names=[column list])
plt.show()
```

# XGBoost

```python
# can also be used with GridSearchCV
xgb_clf = xgb.XGBClassifier(
    max_depth=5,
    n_estimators=1000,
    learning_rate=0.3,
    n_jobs=-1
)
xgb.fit(X, y)

xgb.plot_importance(xgb_clf)
```

# Datetime

```python
# ephoch calculations
desired_date = datetime(year, month, day)
epoc = datetime.fromtimestamp(0)
days_since = (desired_date - epoch).days

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
    infer_datetime_format=True
    #
    # format="%Y-%M-%D
)

now.strftime(format) # format the date but will be string

# integer representation of the date
df.dt.date
df.dt.year
df.dt.month
df.dt.dayofweek
df.dt.time
df.dt.hour
df.dt.minute
df.dt.second

df.to_timedelta()
# "D", "W", "H", "T", "s"
# can use these in resampling but not in timedelta
# "ME", "QE", "YE"

# forward filling dates
dates.ffill()
# back filling dates
dates.bfill()
# interpolate
dates.astype("float64")
dates.interpolate()

# differencing downsampling
dates.diff()
dates.resample("M", on="column").sum()
dates.resample("M").transform("sum")

# upsampling - just resample on a higher frequency the add in data
df.resample("M", on="column").mean()
df.interpolate(method=["linear", "spline"], order=2) # order required for spline, spline is polynomial curve

```

# strings

```python
series.str.strip()
series.str.lstrip()
seires.str.rstrip()
series.str.upper()
series.str.lower()
series.str.slice(start:stop:step)
series.str.count("string") # all instances of string
series.str.contains("string") # true if found
series.str.replace("what", "with")
series.str.split("delimiter", exapnd=True) # return df with splits as columns
series.str.len()
series.str.startswith("string")
series.str.endswith("string")
```

# sql database

```python
import numpy as np
import pandas as pd

from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

driver = "ODBC Driver 17 for SQL Server"
server = "server name"
db_name = "database name"
connection_string = (
    f"Driver={driver};Server={server};Database={db_name};trusted_connection=yes"
)
connection_url = URL.create(
    "mssql+pyodbc, query={"odbc_connect": connection_string}
)

query = text(
    """
    SELECT *
    FROM some_table
    """
)

engine = create_engine(connection_url, echo=False)
Session = sessionmaker(bind=engine)

with Session.begin() as session:
    result = session.execute(query)
    rows = result.fetchall()
    df = pd.DataFrame(rows)

```

# sftp

```python
import paramiko

sftp_user = "username"
sftp_pass = "pass"
sftp_port = 22 # default port
sftp_host = "host url"
df_csv = df.to_csv(sep="|")

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(sftp_host, username=sftp_user, password=sftp_pass)

ftp = ssh.open_sftp()
file = ftp.file("file/path", "w", -1)

bytes_object - df_csv.encode()

file.write(bytes_object)

ftp.close()
ssh.close()
```

# Time series 

```python
#read csv to set index as date
pd.read_csv(
    filepath,
    parse_date=[0], # column index
    index=0 # column index
).squeeze("columns") # will make it a series

time_series = df["value column of series"]
figure = plot_acf(time_series)
figure = plot_pacf(time_series)
plt.show()
```

```python
df = df.assign(
    month=df["date"].dt.month,
    day=df["date"].dt.day,
    year=df["date"].dt.year,
    lag1=df["births"].shift(1),
    lag2=df["births"].shift(2),
    last_year=df["births"].shift(365),
    roll_mean=df["births"].rolling(window=2, center=False).mean(), # can set the window label as the center or the right edge(false)
    rolling_max=df["births"].rolling(window=3).max(),
    expand_max=df["births"].expanding().max()
)
```

```python
# observerd, trend, seasonal, redisduals
result = seasonal_decompose(df["column"], model=["additive", "multiplicative"]) # series must have a datetime index
result.plot()
```

```python
# differencing (difference between time slot and its lagged value)
df["column"].diff(periods=1)
```

```python
# test, train, split
train_size = int(df.shape[0] * 0.8)
train_set = df.iloc(0: train_size, :)
test_set = df.iloc(train_size: 0, :)
```

```python
# naive forecast
# just use the shift(1) value as the prediction
```

```python
# auto regressive model
# create train and test
ar_model = AR(train)
ar_fit = ar_model.fit()
# lag values in model
ar_fit.k_ar
# coefficients for lagged variables
ar_fit.params
# if the data set is just one long set
predictions = ar_fit.predict(start=len(train), end=len(train) + len(test) - 1)
```

```python
# ARIMA
model = ARIMA(df["column"], order=(p,d,q))
model_fit = model.fit()
model.summary()
model.resid
```

```python
# walk forward validation
data = train
predict = []
for t in test:
    model = ARIMA(data, order=(p, d, q)) # or model = AR()
    model_fit = model.fit()
    y = model_fit.forecast()
    predict.append(y[0][0])
    data = np.append(data, t)
    data = pd.series(data)
```

```python
# SARIMAX in python (p, d, q)(P, D, Q, m) where m is the number of time steps for a single seasonal period (how many lags essentialy)
model = SARIMAX(df["column"], order=(p, d, q), season_order=(P, D, Q, m))
model_fit = model.fit()
residual = model_fit.resid
ourpute = model_fit.forecast()
```

# Feature Selection

### constant features
```python
sel = VarianceThreshold(threshold=0) # by default it finds 0 variance features
se.fit(X_train) # fit find the features with 0 variance
```
```python
constant = [col for col in X.columns if X[col].std() == 0]

# to also handle categorical first cast as object
X = X.astype("O")
constant = [col for col in X.columns if X[col].nunique() == 1]
```

### Quasi constant features

```python
quasi_constant_feat = []

for feat in X_train.columns:
    predominant = (
        (X_train[feat].value_counts() / float(len(X_train)))
        .sort_values(ascending=False)
        .values[0]
    )
    if predominant > 0.999:
        quasi_constant_feat.append(feat)
```

```python
sel = DropConstantFeatures(tol=0.998, variables=None, missing_values="raise")
sel.fit(X_train)
sel.feature_to_drop_
X_train = sel.transform(X_train)
```

### duplicated features
```python
duplicated_feat_pairs = {}
_duplicated_feat = []

for i in range(len(X_train.columns)):
    feat_1 = X_train.columns[i]
    if feat_1 not in _duplicated_feat:
        duplicated_feat_pairs[feat_1] = []
        for feat_2 in X_train.columns[i + 1:]:
            if X_train[feat_1].equals(X_train[feat_2]):
                duplicated_feat_pairs[feat_1].append(feat_2)
                _duplicated_feat.append(feat_2)
```

### Correlated Features

```python
sel = DropCorrelatedFeatures(
    threshold=0.8,
    method="pearson",
    missin_values="ignore"
)
sel.fit(X_train)
X_train = sel.transform(X_train)
```

```python
rf_clf = RandomForestClassifier(
    n_estimators=100,
    random_state=0,
    n_jobs=-1
)
sel = SmartCorrelatedSelection(
    variables=None,
    method="pearson",
    threshold=0.8,
    missing_values="ignore",
    selection_method="model_performance",
    estimator=rf_clf,
    scoring="roc_auc",
    cv=3
)
sel.fit(X_train, y_train)
group = sel.correlated_feature_sets_[i]
X_train[group].std()

sel.features_to_drop_
```



# pipeline

```python
pipe = Pipeline(
    [
        ("constant", DropConstantFeature(tol=.998)),
        ("duplicated", DropDuplicateFeatures())
    ]
)

pipe.fit(X_train)

X_train = pipe.transform(X_train)
X_test = pipe.transform(X_test)

len(pipe.named_stesps["constant"].features_to_drop_)
pipe.named_steps["duplicated"].features_to_drop_
```

correlated features pipline

```python
pipe = Pipeline([
    ("constant", DropConstantFeatures(tol=0.998)),
    ("duplicated", DropDuplicatedFeatures()),
    ("correlation", SmartCorrelatedSelection(
        method="pearson",
        threshold=0.8,
        missing_values="ignore",
        selection_method="model_performance", # or "variance"
        estimator=rf_clf,
        scoring="roc_auc",
        cv=3
    ))
])

pipe.fit(X_train, y_train)
X_train = pipe.transfrom(X_train)
X_test = pipe.transform(X_test)

def run_logistic(X_train, X_test, y_train, y_test):
    
    # function to train and test the performance of logistic regression
    logit = LogisticRegression(random_state=44, max_iter=500)
    logit.fit(X_train, y_train)
    print('Train set')
    pred = logit.predict_proba(X_train)
    print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
    print('Test set')
    pred = logit.predict_proba(X_test)
    print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


scaler = StandardScaler().fit(X_train)

run_logistic(scaler.transform(X_train),
             scaler.transform(X_test),
                  y_train, y_test)
```

# mutual information

```python
mi = mutual_info_classif(X_train, y_train)
mi = pd.Series(mi)
mi.index = X_train.columns
mi.sort_values(ascending=False)
```

```python
sel = SelectKBest(mutual_info_classif, k=10).fit(X_train, y_train)
X_train.columns[sel.get_support()]
```
# chi square

```python
c = pd.crosstab(y_train, X_train["columns"])
chi2_contingency(c)

chi_ls = []

for feature in X_train.columns:
    c = pd.crosstab(y_train, X_train[feature])
    p_value = chi2_contingency(c)[1]
    chi_ls.append(p_value)

# select features with lowest p-values
```

# ANOVA p-values

```python
univariate = f_classif(X_train, y_train)
p-values = pd.Series(univariate[1])
uni_index = X_train.columns

sel = SelectKBest(f_classif, k=10).fit(X_train, y_train)
X_train.columns[sel.get_support()]
X_train = sel.transform(X_train)
```

# Select by single feature importance

```python
sel = SelectBySingleFeaturePerformanc(
    variables=None,
    estimator=rf_clf,
    scoring="roc_auc",
    cv=3,
    threshold=0.5
)
sel.fit(X_train, y_train)
sel.feature_performance_

X_train = sel.transform(X_train)
X_test = sel.transform(X_test)
```

# select by target mean performance

```python
sel = SelectByTargetMeanPerformance(
    variables=None, # automatically finds categorical and numerical variables
    scoring="roc_auc", # the metric to evaluate performance
    threshold=0.6, # the threshold for feature selection, 
    bins=3, # the number of intervals to discretise the numerical variables
    strategy="equal_frequency", # whether the intervals should be of equal size or equal number of observations
    cv=2,# cross validation
    regression=False, # whether this is regression or classification
)

sel.fit(X_train, y_train)

sel.feature_performance_

X_train = sel.transform(X_train)
X_test = sel.transform(X_test)
```

# step forward Feature selection


Can be done with mlxtend and scikit-learn
```python
sfs = SFS(
    estimator=rf_clf,
    k_features=10,
    forward=True, # forward = False is backwards selection
    floating=False,
    scoring="roc_auc",
    cv=3
)

selected_features = X_train.columns[list(sfs.k_feature_idx_)]
```

# exhaustive feature selection

```python
efs = EFS(
    estimator=rf_clf,
    min_features=1,
    max_features=4,
    scoring="roc_auc",
    cv=2
)
efs.best_idx_
selected_features = X_train.columns[list(efs.best_idx_)]
```

# from coefficients

```python
sel = SelectFromModel(
    estimator=logistic_regression,
    threshold="mean"
)
sel.fit(X_train, y_train) # careful to scale these features
sel.get_support()
selected_feat = X_train.columns[sel.get_support()]
sel.estimator_.coef_
```

```python
sel = SelectFromModel(
    estimator=linear_regression
)
sel.fit(X_train, y_train) # careful to scale these features
sel.get_support()
selected_feat = X_train.columns[sel.get_support()]
sel.estimator_.coef_
```


```python
sel = SelectFromModel(
    estimator=lasso
)
sel.fit(X_train, y_train) # careful to scale these features
sel.get_support()
selected_feat = X_train.columns[sel.get_support()]
sel.estimator_.coef_
```
# select feature recursively

```python
sel = RFE(
    estimator=RandomForestClassifier(),
    n_features_to_select=10,
)
sel.fit(X_train, y_train) # careful to scale these features
sel.get_support()
selected_feat = X_train.columns[sel.get_support()]
sel.estimator_.coef_
```

# feature select by shuffling

```python
sel = SelectByShuffling(
    estimator=RandomForestClassifier(),
    scoring="roc_auc",
    threshold=0.05,
    cv=3,
    random_state=0
)
sel.fit(X_train, y_train)
sel.initial_model_performance_
sel.performance_drifts_
X_train = sel.transform(X_train)
X_test = sel.transform(X_test)
```

```python
sel = RecursiveFeatureElimination(
    variables=None, # automatically evaluate all numerical variables
    estimator = model, # the ML model
    scoring = 'roc_auc', # the metric we want to evalute
    threshold = 0.0005, # the maximum performance drop allowed to remove a feature
    cv=2, # cross-validation
)

# this may take quite a while, because
# we are building a lot of models with cross-validation
sel.fit(X_train, y_train)
# performance of model trained using all features

sel.initial_model_performance_
X_train = sel.transform(X_train)
X_test = sel.transform(X_test)

X_train.shape, X_test.shape
```
