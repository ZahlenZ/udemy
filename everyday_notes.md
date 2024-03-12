# Libraries

```python
import matplotlib.gridspec as gricspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

from datetime import datetime
from numpy.random import default_rng
from sklearn import tree
from sklearn.ensemble import(
    BaggiingClassifier,
    RandomForestClassifier,
    GradientBoosting,
    AdaBoostClassifier
)
from sklearn.metrics import (
    accuracy_score,
    confustion_matrix,
    mean_squared_error,
    r2_score
)
from skleanr.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.preproccessing import StandardScaler
from sklearn.tree import plot_tree
```


# pandas and numpy

```python
# Importing Data
pd.read_csv() 

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

# differencing
dates.diff()
dates.resample("M").sum()
dates.resample("M").transform("sum")
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

