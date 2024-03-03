# Object Oriented
bets approach with the most customization

will automatically use the index as the x axis and will plot each column as a series, however best to seperate them out for further customization.

formatting options
|option|call|
|---|---|
|figure title| fig.suptitle()|
|char title|ax.set_title()|
|x axis label|ax.set_xlabel()|
|y axis label|ax.set_ylabel()|
|legen|ax.legend()|
|x axis limit|ax.set_xlim()|
|y axis limit|ax.set_ylim()|
|x axis ticks|ax.set_xticks()|
|y axis ticks|ax.set_yticks()|
|vertical line| ax.axvline()|
|horizontal line| ax.axhile()|
|text|ax.text()|
|spine(borders)|ax.spines["side"]|

```python
from matplotlib import colormaps
list(colormaps)
```

```python
import matplotlib.pyplot as plt
```

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

legend locations - by default will try to find the most white space but with loc can choose

loc method
upper right, upper left, upper center, lower right, lower left, lower center, center right, center left, center

bbox_to_anchor - x, y coordinates with the bottom left being 0,0 and runs to 1,1 if you go outside the 1, 1 it will push it outside of the plot area

units that we set in ax.set_xlim() will be the same units as the column of data, so if the data in the xaxis is a date then it is number of days from the epoch (january 1, 1970)

### Doing some epoch time calculations

```python
desired_date = datetime(year, month, day)
epoch = datetime.fromtimestamp(0)
days_since = (desired_date - epoch).days
```

```python
# another method
datetime(2018, 3, 21) - datetime(1970, 1, 1)
```
a way to programatically change some things
```python
colors = ["orange", "grey", "grey", "grey"]
ls = ["-", "--", ":", "-."]

fig, ax = plt.subplots(figsize=(8, 4))

for i, column in enumerate(df.columns):
    ax.plot(
        df[column],
        color=colors[i],
        ls=ls[i]
    )

ax.text(
    x,
    y,
    "...while prices elsewhere \n in california keep rising" # throw in the \n to get a new line in text
)

then the rest of your stuff
```

### Lollipop charts

```python
fig, ax = plt.subplots()

# need something to use for the x value or just df.index
# need something to use for the y value or just df[some column]
ax.stem(
    x,
    y,
    basefmt=" ",
    label="appear in legend"
)
```

ax.plot is a linechart
pivot tabular data to turn each unique series into a dataframe column and se tthe datetime as the index

### stacked line chart
stacks add from the bottom up

```python
fig, ax = plt.subplots()

ax.stackplot(
    df.index,
    df[column], # this one is on bottom
    df[column], # then this column
)
```

### dual axis charts

```python
fig, ax = plt.subplots()

ax.plot(normal plot shit)

ax2 = ax.twinx()

ax2.plot(
    normal plot shit
)
ax2.set_ylabel("this will go on the right")

fig.legend() # set the legend on figure scope to pick up both ax
```

### bar charts
ax.bar(category labels, bar heights, formatting options)
hot tip: use .groupby() and .agg() to aggregate your data by category and push the labels into the index
use seaborn for grouped bar charts

```python
ax.bar(category labels, barheights, formatting)
ax.barh(category labels, bar "length", formatting)

# to flip the order use the index method

category[::-1]
data[::-1]
```

### stacked bar charts
values for bottom bar are the reference for the next bar instead of the axis

```python
ax.bar(
    df.index,
    df[value column],
    label="first label"
)
ax.bar(
    df.index,
    df[value column],
    label="second label",
    bottom=df[previous value column]
)
```

to create a 100% stacked bar chart convert dataframe to row-level percentages

```python
df = df.apply(lambda x: x * 100 / sum(x), axis=1)
```

### grouped bar charts

when doing the width you have to think about how many bars you have in each category, and make sure that the width * bar count less than 1 or they will overlap
```python
width = .35
x = np.arrange(3) # number of categories

ax.bar(
    x-width / 2, # shift bars to left
    df[value column],
    width=width,
    label="text"
)
ax.bar(
    x + width / 2, # shift bars to right
    df[value column 2],
    width=width,
    label="text",
    color="orange", 
    alpha=.3 # same argument as ggplot, sets transparency
)

ax.set_xticks(x)
ax.set_xticklabels(df.index)
```

### Combo Charts

use the first ax for one of the plots and ax2 for the other plot

### Pie Charts

```python
ax.pie(
    series_values,
    labels=,
    startangle=90, # sets the fist slice to start on the top
    autopct="%.Of%%",
    pctdistance=,
    explode=
)
```

### donus

```python
fig, ax = plt.subplots()
ax.pie(
    series_values,
    labels=,
    startangle=90,
    autopct="%Of%%",
    pctdistance=.85 # shifts the labels out
)

donut_hole = plt.Circle((0, 0), .70, fc="white")
fig = plt.gcf()

fig.gca().add_artist(donut_hole)
ax.set_title("nifty title")
```

### scatter plots

```python
ax.scatter(
    x-axis-series,
    y-axis-series,
    size=,
    alpha=
)
```

### histogram

set a more transparent alpha and plot multiple histograms on the same plot

```python
ax.hist(
    series,
    density=False, # if set to true use relative frequencies, percent of total
    alpha=,
    bins=
)
