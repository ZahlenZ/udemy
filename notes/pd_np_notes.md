# Thing to remember
(row, column)
axis=0 is across the rows
axis=1 is down the columns

### Operators & Methods
|Description|Python|Pandas|
|---|---|---|
|Equal|==|.eq()|
|Not Equal|!=|.ne()|
|less than or equal|<=|.le()|
|less than|< |.lt()|
|greater than or equal| >= |.ge()|
|greater than| >|.gt()|
|membership test| in | .isin() |
|inverse membership| not in| ~.isin()|
|or|  \| | 
|and|  & | 
|addition| +| .add() |
|subtraction| - | .sub() |
|multiplication| * | .mul() |
|division| / | .div() .truediv() |
|flood division| //| .floordiv() |
|modulo| % |.mod()|
|exponent| ** | .pow()|

### Text data
string_series.str.method()
|Method|Descrtiption|
|---|---|
|.str.strip(), .lstrip(), .rstrip()| removes all leading and or trailing characters|
|.str.upper(), .lower() | converts to upper or lower|
|.str.slice(start:stop:step)| applies a slice to the strings|
|.str.count("string")|count all instances of given string|
|.str.contains("string")|true if given string is found, false if not|
|.str.replace("a", "b")| replace instances of "a" with "b"|
|.str.split("delimiter", "expand=True)|splits string on delimiter, return DF with series for each split|
|.str.len()| length of each string|
|.str.startswith("string"), .endswith("string")| returns true is string starts with or ends with|

### Aggregate 
|Method|Description|
|---|---|
|.count()|returns the number of item|
|.first(), .last()|returns the first or last item|
|.mean(), .median() | mean or median|
|.min(), .max() | min or max|
|.argmax(), .argmin()| index for the smalles or largest|
|.std(), .var() | std deviation or variance |
|.mad()| mean absolute deviation|
|.prod| product of all the items|
|.sum() | sum|
|.quantile | specified percentile or list of percentiles|
|.unique() | unique items|
|.nunique() | number of unique items|
|.value_counts() | unique items and their frequencies|


```python
# membership examples
df.index.isin(list)
df.loc[df["column"].isin(list)]
```

# Random Number Generation
```python
from numpy.random import default_rng
# rng class
rng = default_rng(seed=289)
# random numbers uniform distribution
rng.random(n)
# random numbers normal distribution
rng.normal(mean, stdev, n)
# random integers
rng.integers(low, high, n) # high exclusive (no 10's in here)

```

# Array
```python
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
# array functions
np.median(array)
np.percentile(array, n) # nth percentile
np.unique(array)
```

# series

### Datatypes
|DataType|Pythons|
|---|---|
|int|int8, 16, 32, 64|
|float|float32, 64|
|object|any python object|
|string|only text|
|category|categorical data|
|datetime64|moment in time|
|timedelta|duration between 2 dates|
|period|span of time|

### Some Operations
```python
# convert data type
df.astype(
    {
        "column": "datatype",
        "columnd": "datatype"
    }
)
## indexing
# access index
df.index
# set index one way
df.index = ["something", "something"]
# reset index
df.reset_index(drop=True) # drop false to keep the old index in a column
## slicing
# access values by their positional index
df.iloc[row postion/logical, column position]
# access values by their labels
df.loc[row postion/logical, column labels]
## sorting
df.sort_values("columnname", ascending=True)
```

### NaN values
Pandas treats NaN values as a float, so they can be used in vectorized operations
nan is numpy
np.nan
pandas missing data time, NA, <NA> stored as an integer
pd.NA

```python
# replace NaN with something
series.add(2, fill_value=0)
# finding it
series.isna()
series.isna().sum()
# dataframe na count by column
df.isna().sum()
# value_counts
df["column"].value_counts(dropna=False)
# deal with missing
df.dropna() # removes NaN values from series or df
df.fillna("something") # replaces NaN values
```

### Apply Method
Non vectorized but lets functions be applied to series or df

```python
def discount(price):
    if price > 20:
        return round(price * 0,9, 2)
    return price

# modify all values in the series
series.apply(discount)
# one off
series.apply(lambda x: round(x * 0.9 if x > 20 else x))
```

### Where Method
return series values based on logical condition

```python
# replace if false, keep original if true
df.where(
    logical test,
    value if False,
    inplace = False
)
```

### Dataframes

```python
df.head(n)
df.tail(n)
df.sample(n)
df.info(show_counts=False, memory_usage="deep")
df.describe()
```

```python
df.iloc[logical/row, column index]
df.loc[logical/row, column label] # returns series
df.loc[[]] # returns dataframe
```

```python
# drop columns
df.drop([list of columns], axis=1, inplace=False)
# drop rows by index
df.drop([slice or list], axis = 0, inplace=False)
# identified duplicates
df.duplicated(subset = [column list])
# drop duplicates
df.drop_duplicates(
    subset = [list of columns],
    keep="last",
    ignore_index=True
)
```

```python
df.fillna({dictionary key = column, value = what to fill with})
df.dropna(subset=[list of columns to look for NA and drop those rows])
```

```python
df.loc[logical on df["column"], [column list]]
# use .query to use SQL like syntax to filter
df.query("family in ['list', 'coluns'] and sales > 0)

avg_sales = something
df.query("family in ['list', 'columns'] and sales > @avg_sales)
```

```python
# by default sorts index by row, axis=1 to sort the columns
df.sort_index(axis=0, inplace=False, ascending=True)
# by value
df.sort_values(["list", "columns"], ascending=[True, False])
```

```python
# one way to change column names
df.columns = ["list", "columns"]
# another way
df.rename(
    columns =
        {
            "old name": "new name",
        },
    inplace=False
)
# reorder columns
df.reindex(labels = ["new", "column", "order], axis=1)
```

```python
# select function
condition = [
    list of,
    some logicals
]
choices = [
    outcome of,
    logicals
]
df["new column"] = np.select(conditions, choices, default="some string")
```

```python
# map method
# pass diction with existing values as the keys, and new values as values
mapping_dict = dict
df["new column"] = df["column"].map(mapping_dict)
```

```python
# assign method what to make it can be complex calculations
df.assign(
    column_name = what to make it,
    column_name2 = what to make it
)

df = df.assign(
    onpromotion_flag = df["onpromotion"] > 0,
    onpromotion_ration = df["sales"] / df["onpromotion"],
    sales_onprom_target = lambda x: x["onpromotion_ratio"] > 100
).query("sales_onprom_target == True")
```

```python
df.astype({dict key=column, value=dtype})
```

```python
df.loc[:, ["column", "column"]].sample(100).sum().round(2)

df.groupby(
    ["column", "another column"],
    as_index=False # will make sure the index is repeated and not grouped
)[["column"]].sum()

df.groupby(
    ["column", "another column"]
).agg({"Call it this": ["give", "these"]})

# to avoid multi index columns
(
    df
    .groupby(["column1", "column2"])
    .agg(
        sales_sum=("column", "sum"),
        sales_avg=("sales", "mean")
    )
)
```

### Multi Index DataFrame

index is stored as a list of tuples, with an item for each layer of the index.
access rows by outer index

```python
# filter down to an outer index
df.loc["outer index"]
# access single value
df.loc[("outer index", "inner index"), :]
# reset the index
df.reset_index()
# swap the index level
df.swaplevel()
# drop an index level
df.droplevel("level to drop")
```

### Transform
The .transform() method can be used to perform aggregations without reshaping (reducing rows) transform will add the aggregation to each row

```python
df.assign(
    store_sales = ( # create a new row
        df
        .groupby("store_nbr")["sales"] # group by store_nbr
        .transform("sum") # aggregate this
    )
)
```

### Pivot Tables
 Unlike excel pandas pivot tables don't have "filter" but you can filter your dataframe before pivoting to return a filtered pivot table

 can also pass dictionary to aggfunc to do different aggregations 

 {"sales": ["sum", "mean"], "onpromotion": "max"}
```python
 df.pivot_table(
    index="column", # category on the left of pivot
    columns="column", # category on the top of pivot
    values="column", # column the values come from
    aggfunc=("sum", "mean", "max"), # what aggregation to use
    margins=False # returns row and column totals when True
 )
 # axis=None will heatmap highest values across whole pivot
 # axis=1 will heatmap highest values in each row (across the columns)
 # axis=0 will heatmap highest values in each columng (down the rows)
 df.pivot_table().style.background_gradient(cmap="", axis = None)
 ```

 cmap values
 magma, inferno, plasma, viridis, cividis, twilight, twilight_shifted, turbo, Blues, BrBG, BuGn, BuPu, CMRmap, GnBu, Greens, Greys, OrRd, Oranges, PRGn, PiYG, PuBu, PuBuGn, PuOr, PuRd, Purples, RdBu, RdGy, RdPu, RdYlBu, RdYlGn, Reds, Spectral, Wistia, YlGn, YlGnBu, YlOrBr, YlOrRd, afmhot, autumn, binary, bone, brg, bwr, cool, coolwarm, copper, cubehelix, flag, gist_earth, gist_gray, gist_heat, gist_ncar, gist_rainbow, gist_stern, gist_yarg, gnuplot, gnuplot2, gray, hot, hsv, jet, nipy_spectral, ocean, pink, prism, rainbow, seismic, spring, summer, terrain, winter, Accent, Dark2, Paired, Pastel1, Pastel2, Set1, Set2, Set3, tab10, tab20, tab20b, tab20c, grey, gist_grey, gist_yerg, Grays, magma_r, inferno_r, plasma_r, viridis_r, cividis_r, twilight_r, twilight_shifted_r, turbo_r, Blues_r, BrBG_r, BuGn_r, BuPu_r, CMRmap_r, GnBu_r, Greens_r, Greys_r, OrRd_r, Oranges_r, PRGn_r, PiYG_r, PuBu_r, PuBuGn_r, PuOr_r, PuRd_r, Purples_r, RdBu_r, RdGy_r, RdPu_r, RdYlBu_r, RdYlGn_r, Reds_r, Spectral_r, Wistia_r, YlGn_r, YlGnBu_r, YlOrBr_r, YlOrRd_r, afmhot_r, autumn_r, binary_r, bone_r, brg_r, bwr_r, cool_r, coolwarm_r, copper_r, cubehelix_r, flag_r, gist_earth_r, gist_gray_r, gist_heat_r, gist_ncar_r, gist_rainbow_r, gist_stern_r, gist_yarg_r, gnuplot_r, gnuplot2_r, gray_r, hot_r, hsv_r, jet_r, nipy_spectral_r, ocean_r, pink_r, prism_r, rainbow_r, seismic_r, spring_r, summer_r, terrain_r, winter_r, Accent_r, Dark2_r, Paired_r, Pastel1_r, Pastel2_r, Set1_r, Set2_r, Set3_r, tab10_r, tab20_r, tab20b_r, tab20c_r, rocket, rocket_r, mako, mako_r, icefire, icefire_r, vlag, vlag_r, flare, flare_r, crest, crest_r

### Melting Dataframes
.melt()

```python
df.melt(
    id_vars="column",
    value_vars=[list],
    var_name="name",
    value_name="name
)
```
id_vars is the column that will be repeated down the new dataframe
value_vars is the columns that will be transfered/melted

each unique id_var has a value for the value_vars the melted table will have unique combinations of the id_var and value_var with the associated value.

# Dates and Times
pandas will treat dates as "objects" until converted to datetime64

```python
from datetime import datetime
```

```python
# current time
now = datetime.now()
```

If there is a missing value in the datetime column the astype method and will have error
Better to use pd.to_datetime() function

when using coerce thing that can't be converted will become NaT (not a time)

```python
df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
    infer_datetime_format=True,
    # format="%Y-%M-%D can use this to declare the format, but best to infer
)
```

for more strftime(fmt) go to https://pandas.pydata.org/docs/reference/api/pandas.Period.strftime.html

```python
now.strftime(format) # this will format it but it will be strings

df.assign(date=df["date"].dt.strftime(format))

# all return an integer representation of the date
df.dt.date
df.dt.year
df.dt.month
df.dt.dayofweek
df.dt.time
df.dt.hour
df.dt.minute
df.dt.second

df = df.assign(
    year=df["date"].dt.year
)
```

### Time Delta
.to_timedelta()
"D" - Day
"W" - Week
"H" - Hour
"T" - minute
"S" - second

can use these is resampling but not in timedelta arithmetic
"M" - month end
"Q" - quarter end
"Y" - year end

```python
pd.to_timedelta(5, unit="D")
pd.to_timedelta(5, unit="W")
# can't do this with years
```

### DateTimes as index

```python
df.loc[df["date"].dt.year == 2022]

df.set_index("date").loc["2023"]
```

### Missing Time Series 

time series data allows for methods for fixing missing data beyond .fillna()

```python
# forward filling pulls from previous date
dates.ffill()
# back filling pulls from netx date
dates.bfill()
# interpolate
dates = dates.astype("float64")
dates.interpolate()
```

### Shift Series
.shift()

```python
# will shift down by n row
# positive for forward, negative for backward
series.shit(n)

# example growth over time
(dates / dates.shit(1)).sub(1).mul(100).round(2)
```

.diff()
```python
# difference in each row - value prior thing lag
dates.diff() 
```
### Aggregation & Resampling

```python
# calculating sum of each calendar month
dates.resample("M").sum()
# transform() to keep the original index and apply group level aggregations
dates.resample("M").transform("sum")
```

### Rolling aggregation

```python
# average of 2 points, avg of current row and previous row
dates.rolling(2).mean()
```

# Importing and Exporting Data

```python
read_csv(
    file="path/name.csv",
    sep=",",
    header=0,
    nrows=5, # number of rows to bring in
    skiprows=[index of rows to skip], # can also do a lambda function
    na_values=[list of things to treat as na], # will replace with NaN
    names=["list", "of", "override", "columnnames"],
    index_col="column",
    usecols=["columns to keep"],
    dtype={key = column value = datatype},
    parse_dates=True,
    infter_datetime_format=True
)
```
converters, apply functions to columns as we read them in

```python
read_csv(
    coverters = {
        "sales": lambda x: x if x not in ["missing", "."] else 0,
        "sales.1": lambda x: f"${x}"
    }
)
```

Reading in text files be careful of what the seperater is and use read_csv()

```python
# reading in from excel
# no sheet name just reads in the first
# sheet_name = 1 reads in the second sheet
pd.read_excel("monthly_sales.xlsx", sheet_name=1)

pd.concat(
    pd.read_excel("monthly.xlsx", sheet_name=None),
    ignore_index=True
)

# exporting
# to_csv and to_excel
df.to_csv("file_name.csv", sep="thing")
df.to_excel("workbook.xlsx", sheet_name="OctoberSales")
```

### working with sql database

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

### Other file formats to right to

|format|ReadFunction|WriteFunction|Description|
|---|---|---|---|
|JSON|read_json|to_json||
|Feather|read_feather|to_feather|new, read/write/and store DFeffeciently|
|HTLM|read_html|to_html||
|Pickle|read_pickle|to_pickle|serialized storage allows quick reproductions|
|python dictionary|pd.DataFrame|to_dict||


Interesting, get data from webpage, exmaple with wikipedia

```python
url = ""
# table index to read if there are multiple (first is 0)
df = pd.read_html(url)[0]
```

# Joining and appending
### Vertical
```python
# stack the dataframes
pd.concat([df1, df2, df3])
```
### Join
can join on multiple columns if that is how you make a unique id
```python
left_df.merge(
    right_df,
    how,
    left_on,
    right_on
)
```

### join types

left join _ all records in left with matching in right
inner join - records exist in Both Tables excludes unmatched
outer join - all records from both tables, including non-matching
