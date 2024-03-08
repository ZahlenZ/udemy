# seaborn

seaborn will aggregate automatically, unlike matplotlib

```python
import seaborn as sns

sns.lineplot(
    x="columnname",
    y="columnname",
    data=df,
    estimator=sum, #default is np.mean
    hue="column" # will seperate out the series if we are aggregating legend auto added
    palette="husl" # will change the colormap option: viridis
    ci=None,
    ls="",
    color=""
)

sns.despine() #removes top and right borders by default
```

hue breaks a category into another category

### barchats and histograms

Note that seaborn automatically aggregates the data for the plot, using unique category values as the lables for the bars, the mean of each category for the bar length, and the column headers as the axis labels

to create horizontal barchart just flip the x and y
sns.barplot(
    x,
    y,
    data
)

sns.histplot(
    x=,
    data=df,
    bins=,
    kde=False set to true will give a smooth curve, density function
)

### boxplots and violin plots

```python
# pass in x and y to get seperate box plots based on the variables
# one for the numerical, and one for the categorical
sns.boxplot(
    x=,
    y=,
    data=
)

# by default has the box plot in the middle of it
sns.violinplot(
    x=,
    y=,
    data=
)
```

### Linear Relationship plots

```python

sns.scatterplot(
    x,
    y,
    data
)
# scatterplot with fitted regresion line
sns.regplot(
    x,
    y,
    data
)
# extensino of regplot
# with row and col you get a matrix for each combination of row and column
sns.lmplot(
    x,
    y,
    hue, # get a regression line for every level of teh categorical variable
    row, # another categorical
    col, # another categorical
    data
)
# scatter plot and distribution of each variable
sns.jointplot(
    x,
    y,
    kind, #play with kde "density" of data also reg and some others
    data
)
# scatter plot matrix comparing multiple variables and shows the distribution for each one along the diagonal
sns.pairplot(
    cols,
    hue="",
    palette="",
    corner=True # corner true will only show the lower left diagonal and not the inverse of it in the upper diagonal
    diag_kind="kde" # "auto" "hist" "kde" None
)
```

### heatmaps

```python
sns.heatmap(
    df_pivot,
    annot=True #provide labels on cells
    format=g, # formats the numbers
    cmap="RdYlGn"
)
```

df.corr() grabs all numerical columns and calculates the correlation between them

### facet grids

the number of subplots will be the same as the number of categories in column

```python
g = sns.FacetGrid(
    data,
    column,
    columnwrap
)
g.map_dataframe(sns.histplot, x="price")
```

### Matplotlib integration

can use all of the matplotlib stuff in seaborn

example

```python
fix, ax = plt.subplots()

sns.set_style("darkgrid")

sns.barplot(
    data=,
    x=,
    y=
)

ax.set_title()
ax.set_xlabel()
ax.set_ylabel()
```
all seaborn objects have the ax argument so they can also be incorporated into the subplots in matplotlib

```python
fig, ax = plt.subplots(2,1,)

sns.set_style("darkgrid")
sns.barplot(
    data,
    x,
    y,
    data,
    ax=
)

ax[1].hist()
```

### Density curve

```python
sns.kdeplot(
    data,
    shate=,
    color=,
    label=
)
```

### swarm plots

```python
fig, ax = plt.subplots()
sns.swarmplot(
    data,
    x=,
    y=,
    ax=ax,
    hue="",
    dodge=True
)
```

### correlation matrix view

```python
corrmat = df.corr()
f, ax = plt.subplots(figsize=(n,m))
sns.heatmap(corrmat, vmax=.8, square=True)
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