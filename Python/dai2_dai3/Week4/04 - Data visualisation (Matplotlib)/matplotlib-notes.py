import matplotlib as mlt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('classic')


x = np.linspace(0, 10, 100)

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

plt.show()

#*** Start on IPython Notebook ***#
x = np.linspace(0, 10, 100)

fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--')
plt.show()

# you can save figures to pnf:
fig.savefig('my_figure.png')

# from IPython.display import Image
# Image('my_figure.png')
#*** End on IPython Notebook ***#

print(fig.canvas.get_supported_filetypes())

# Create a figure with two panels which each take two rows, one column:
plt.figure()  # create a plot figure
# create the first of two panels and set current axis
plt.subplot(2, 1, 1) # (rows, columns, panel number)
plt.plot(x, np.sin(x))
# create the second panel and set current axis
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))
plt.show()

# Object oriented interface (instead of using pl.gcf() nor plt.gca(0 get current figure and get current axes we can use OOF:

# First create a grid of plots
# ax will be an array of two Axes objects
fig, ax = plt.subplots(2)

# Call plot() method on the appropriate object
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))
fig.show() # so basically the ax is a list of two Axes objects, we set each object to the data to define the sin and cos values, and then we use the figure to show both axes in one plot combining both Axes' plots

# %matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')

fig = plt.figure()  # white empty canvas ==> container that contains all objects representing axes, graphics, text and labels
ax = plt.axes() # a box with ticks and labels which will contain the plot elements
fig.show()

x = np.linspace(0, 10, 100) # from values 0 to 10, 100 values (in order)
print(x) # mind it's linspace not linespace
ax.plot(x, np.sin(x))
# fig.show()

# same concept but using the pylab interface instead:
plt.plot(x, np.sin(x)) # to be displayed in IPython Notebook

# Figure with multiple lines (Call the plot function multiple times):
x1 = plt.plot(x, np.sin(x))
x2 = plt.plot(x, np.cos(x))
plt.show()

# As you can see we use the plt all times in this file, what's happening is that every time we call it we are re-assign it new values so like re-setting the plot each time, and so basically once plt.show() is called after that we start a new plt which is empty..., because we can only call show on a plt object


#Adjusting line colors and styles
plt.plot(x, np.sin(x - 0), color='blue') # specify color by name
plt.plot(x, np.sin(x + 1), color='g')           # short color code (rgbcmyk)
plt.plot(x, np.sin(x + 2), color='0.75')        # Grayscale between 0 and 1
plt.plot(x, np.sin(x + 3), color='#FFDD44')     # Hex code (RRGGBB from 00 to FF)
plt.plot(x, np.sin(x + 4), color=(1.0,0.2,0.3)) # RGB tuple, values 0 to 1
plt.plot(x, np.sin(x + 5), color='chartreuse') # all HTML color names supported
plt.show()

#Adjusting line style:
# plt.plot(x, x + 0, linestyle='solid')
# plt.plot(x, x + 1, linestyle='dashed')
# plt.plot(x, x + 2, linestyle='dashdot')
# plt.plot(x, x + 3, linestyle='dotted')
# Same as:
# For short, you can use the following codes:

plt.plot(x, x + 4, linestyle='-')  # solid
plt.plot(x, x + 5, linestyle='--') # dashed
plt.plot(x, x + 6, linestyle='-.') # dashdot
plt.plot(x, x + 7, linestyle=':')  # dotted
plt.show()

#Combine adjusting color and line style:
plt.plot(x, x + 0, '-g')  # solid green
plt.plot(x, x + 1, '--c') # dashed cyan
plt.plot(x, x + 2, '-.k') # dashdot black
plt.plot(x, x + 3, ':r')  # dotted red
plt.show()

# adjust plot limits on both axes:
plt.plot(x, np.sin(x))

plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)
plt.show()


#plt.axis([xmin, xmax, ymin, ymax])
plt.plot(x, np.sin(x))
plt.axis([-1, 11, -1.5, 1.5])
plt.show()

#tighten the borders with 'tight' in axis function
plt.plot(x, np.sin(x))
plt.axis('tight') # works in notebook only
plt.show()

plt.plot(x, np.sin(x))
plt.axis('equal') # one unit in x is equal to one unit in y
plt.show()

# Labeling plots:
plt.plot(x, np.sin(x))
plt.title("A Sine Curve")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.show()
# labeling plots with multiple lines via legend/legends
plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.axis('equal')
plt.legend()
plt.show()

"""
MATLAB-stlye functions -> to -> object-oriented methods
-------------------------------------------------------
plt.xlabel()              →       ax.set_xlabel()
plt.ylabel()              →       ax.set_ylabel()
plt.xlim()                →       ax.set_xlim()
plt.ylim()                →       ax.set_ylim()
plt.title()               →       ax.set_title()
"""
# or ax.set() to set all these properties at once in object-oriented plotting:
#%%
ax = plt.axes()
ax.plot(x, np.sin(x))
ax.set(xlim=(0, 10), ylim=(-2, 2),
       xlabel='x', ylabel='sin(x)',
       title='A Simple Plot')
plt.show()


#Scatter plots scatterplots
x = np.linspace(0, 10, 30)
y = np.sin(x)

plt.plot(x, y, 'o', color='black')
plt.show()

# different type of character representation:
rng = np.random.RandomState(0)
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(rng.rand(5), rng.rand(5), marker,
             label="marker='{0}'".format(marker))
plt.legend(numpoints=1)
plt.xlim(0, 1.8)
plt.show()

# joins the dots with dashesliens
plt.plot(x, y, '-ok')
plt.show()

# additional arguments for plot to specify properties
plt.plot(x, y, '-p', color='gray',
         markersize=15, linewidth=4,
         markerfacecolor='white',
         markeredgecolor='gray',
         markeredgewidth=2)
plt.ylim(-1.2, 1.2)
plt.show()


# scatterplots with plt.scatter
plt.scatter(x, y, marker='o')
plt.show()

# #%% md
# The primary difference of **plt.scatter** from **plt.plot** is that it can be used to create scatter plots where the properties of each individual point (size, face color, edge color, etc.) can be individually controlled or mapped to data.
#
# Let's show this by creating a random scatter plot with points of many colors and sizes. In order to better see the overlapping results, we'll also use the alpha keyword to adjust the transparency level:
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)
print("\n\n\nx\n")
print(x)
print("\ny")
print(y)
plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
            cmap='viridis')
plt.colorbar()  # show color scale
plt.show()

print("------------------------------------")
from sklearn.datasets import load_iris
iris = load_iris()
# print(iris)
print()
features = iris.data.T
# print(features)

plt.scatter(features[0], features[1], alpha=0.2,
            s=100*features[3], c=iris.target, cmap='viridis')
print(features[0])
print()
print(100*features[3])
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()

print("-------------------")
print("histograms")
data = np.random.randn(1000)
plt.hist(data)
plt.show()
print(data)

# histogram properties
plt.hist(data, bins=30, density=True, alpha=0.5,
         histtype='stepfilled', color='steelblue',
         edgecolor='black')
#normed=True
plt.show()
print(data)

# multiple histogram data in one histograms
x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)

kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)

plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs)
plt.show()

# If you would like to simply compute the histogram (that is, count the number of points in a given bin) and not display it, the **np.histogram()** function is available:
counts, bin_edges = np.histogram(data, bins=5)
print()
print(counts)
print()

#two dimensional histograms and binnings
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T
plt.hist2d(x, y, bins=30, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')
plt.show()
counts, xedges, yedges = np.histogram2d(x, y, bins=30)
print(counts)

from scipy.stats import gaussian_kde

# fit an array of size [Ndim, Nsamples]
data = np.vstack([x, y])
kde = gaussian_kde(data)

# evaluate on a regular grid
xgrid = np.linspace(-3.5, 3.5, 40)
ygrid = np.linspace(-6, 6, 40)
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

# Plot the result as an image
plt.imshow(Z.reshape(Xgrid.shape),
           origin='lower', aspect='auto',
           extent=[-3.5, 3.5, -6, 6],
           cmap='Blues')
cb = plt.colorbar()
cb.set_label("density")
plt.show()

# Customisation
plt.style.use('classic')

x = np.linspace(0, 10, 100)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), '-b', label='Sine')
ax.plot(x, np.cos(x), '--r', label='Cosine')
ax.axis('equal')
# ax.legend(loc='upper left', frameon=False) # we need to initalise the legend, we can also specify where we want it to be
# ax.legend(frameon=False, loc='lower center', ncol=2)
ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.show()


# personalised
# Customisation
plt.style.use('classic')

x = np.linspace(0, 10, 100)
fig, ax = plt.subplots(2)
ax[0].plot(x, np.sin(x), '-b', label='Sine')
ax[1].plot(x, np.cos(x), '--r', label='Cosine')
# ax.axis('equal')
# ax.legend(loc='upper left', frameon=False) # we need to initalise the legend, we can also specify where we want it to be
# ax.legend(frameon=False, loc='lower center', ncol=2)
fig.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.show()











import matplotlib.pyplot as plt
import numpy as np

# A) Create 2 arrays X and Y (10 random values in each array)
X = np.random.randint(1, 100, 10)  # Random integers between 1 and 20
Y = np.random.randint(1, 100, 10)

# B) Create a figure with 6 x 6 size
plt.figure(figsize=(6, 6))

# C) Plot two lines, each line should be computed by a given array (x-axis) and compute the power of the elements (y-axis)
plt.plot(X, X**2, marker='o', linestyle='-', color='r', label='X squared')  # Red line for X squared
plt.plot(Y, Y**2, marker='o', linestyle='--', color='b', label='Y squared')  # Blue dashed line for Y squared

# D) Add title, xlabel, ylabel, legend to the figure
plt.title('Relationship Between X and Y Squared')
plt.xlabel('X and Y values')
plt.ylabel('Squared values')
plt.legend()  # Add legend

# E) Red color for the first line and Blue for the second one (already done in C)

# F) Dashed line for the second one (already done in C)

# G) Add grid to the plot
plt.grid(True)

# H) Save the figure in PDF format
plt.savefig('squared_plot.pdf', format='pdf')

# Show the plot
plt.show()



x = np.linspace(0, 10, 100)
y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5))
lines = plt.plot(x, y)

# lines is a list of plt.Line2D instances
plt.legend(lines[:2], ['first', 'second'])
plt.show()

## or also same things but clearer:
print(y)
plt.plot(x, y[:, 0], label='first')
plt.plot(x, y[:, 1], label='second')
plt.plot(x, y[:, 2:])
plt.legend(framealpha=1, frameon=True)
plt.show()


cities = pd.read_csv('datasets/Datasets/california_cities.csv')
# print(cities.head())

# Extract the data we're interested in
lat, lon = cities['latd'], cities['longd']
population, area = cities['population_total'], cities['area_total_km2']

# Scatter the points, using size and color but no label
plt.scatter(lon, lat, label=None,
            c=np.log10(population), cmap='viridis',
            s=area, linewidth=0, alpha=0.5)
plt.gca().set_aspect('equal', adjustable='datalim') #plt.axis(aspect='equal')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar(label='log$_{10}$(population)')
plt.clim(3, 7)

# Here we create a legend:
# we'll plot empty lists with the desired size and label
for area in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.3, s=area,
                label=str(area) + ' km$^2$')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')

plt.title('California Cities: Area and Population')
plt.show()




fig, ax = plt.subplots()

lines = []
styles = ['-', '--', '-.', ':']
x = np.linspace(0, 10, 1000)

for i in range(4):
    lines += ax.plot(x, np.sin(x - i * np.pi / 2),
                     styles[i], color='black')
plt.gca().set_aspect('equal', adjustable='datalim') # exaclt the same as: ax.axis('equal')

# specify the lines and labels of the first legend
ax.legend(lines[:2], ['line A', 'line B'],
          loc='upper right', frameon=False)

# Create the second legend and add the artist manually.
from matplotlib.legend import Legend
leg = Legend(ax, lines[2:], ['line C', 'line D'],
             loc='lower right', frameon=False)
ax.add_artist(leg)
plt.show()


# from scipy import np_minversion

print("_------------------------------------------------___________---_")
print("\nVisualisation with seaborn \n\n")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rng = np.random.RandomState(0)
x = np.linspace(0, 10, 500)
y = np.cumsum(rng.randn(500, 6), 0)

# Plot the data with Matplotlib defaults
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left')
plt.show()

# now with seaborn
import seaborn as sns
sns.set() # set style

plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left')
plt.show()


# Histogram with matplotlib.pyplot:
data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])
# print(data)
# Plot histograms with specific colors
colors = ['blue', 'green']  # You can choose any colors you like

for col, color in zip('xy', colors):
    plt.hist(data[col], density=True, alpha=0.5, color=color)
plt.show()

# with seaborn we can use a kernel density to see as smooth estimate of the distribution:
for col, color in zip('xy', colors):
    sns.kdeplot(data[col], fill=True, color=color, label=col)
plt.show()

# Histograms and KDE can be combined using distplot:
# Plot distributions with specified colors
sns.histplot(data['x'], color='blue', kde=True, label='x')
sns.histplot(data['y'], color='green', kde=True, label='y')
plt.show()

# passing a 2dimensional two-dimensional dataset so like a dataframe to a kdeplot:
# Create a 2D KDE plot
sns.kdeplot(x=data['x'], y=data['y'], fill=True)
plt.show()

# Plot joint KDE with the correct syntax
with sns.axes_style('white'):
    sns.jointplot(x="x", y="y", data=data, kind='kde', fill=True)
plt.show()

# hexagonally based histogram:
# Plot joint hexbin plot
with sns.axes_style('white'):
    sns.jointplot(x="x", y="y", data=data, kind='hex')
plt.show()

iris = sns.load_dataset("iris")
print(iris.head())

# visualise multidimensional relationships with sns.pairplot
# Plot pairplot with updated parameters
sns.pairplot(iris, hue='species', height=2.5)
plt.show()

tips = sns.load_dataset('tips')
print(tips.head())


tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']
print(tips['tip_pct'])

grid = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40, 15))
plt.show()


# Define a color palette
palette = {'Male': 'blue', 'Female': 'green'}

# Plot using catplot with updated parameters
with sns.axes_style('ticks'):
    g = sns.catplot(x="day", y="total_bill", hue="sex", data=tips, kind="box", palette=palette)
    g.set_axis_labels("Day", "Total Bill")
plt.show()

sns.jointplot(x="total_bill", y="tip", data=tips, kind='hex')
plt.show()

# Plot joint regression plot
sns.jointplot(x="total_bill", y="tip", data=tips, kind='reg')
plt.show()

planets = sns.load_dataset('planets')
print(planets.head())
# Plot using catplot with updated parameters
g = sns.catplot(x="year", data=planets, aspect=2, kind="count", color='steelblue')

# Customize x-axis tick labels
for ax in g.axes.flat:
    ax.set_xticks(range(0, max(ax.get_xticks()) + 1, 5))  # Adjust tick positions
    ax.set_xticklabels(range(0, max(ax.get_xticks()) + 1, 5))  # Adjust tick labels
plt.show()

# Plot using catplot with updated parameters
g = sns.catplot(
    x="year",
    data=planets,
    aspect=4.0,
    kind='count',
    hue='method',
    order=range(2001, 2015)
)

# Customize y-axis label
g.set_axis_labels("Year", "Number of Planets Discovered")
plt.show()


import datetime

def convert_time(s):
    h, m, s = map(int, s.split(':'))
    return datetime.timedelta(hours=h, minutes=m, seconds=s)

data = pd.read_csv('datasets/Datasets/marathon-data.csv',
                   converters={'split':convert_time, 'final':convert_time})
print(data.head())

data['split_sec'] = data['split'].astype(int) / 1E9
data['final_sec'] = data['final'].astype(int) / 1E9
print(data.head())

with sns.axes_style('white'):
    g = sns.jointplot(x="split_sec", y="final_sec", data=data, kind='hex')
    g.ax_joint.plot(np.linspace(4000, 16000),
                    np.linspace(8000, 32000), ':k')
plt.show()

data['split_frac'] = 1 - 2 * data['split_sec'] / data['final_sec']
print(data.head())

# Plot histogram of 'split_frac'
sns.histplot(data['split_frac'], kde=False)
plt.axvline(0, color="k", linestyle="--")
plt.show()

# to use a condition to count how many rows comply with it:
# better use the sum function in python
sum(data.split_frac < 0)
# instead of:
data[data['split_frac'] < 0].count()


g = sns.PairGrid(data, vars=['age', 'split_sec', 'final_sec', 'split_frac'],
                 hue='gender', palette='RdBu_r')
g.map(plt.scatter, alpha=0.8)
g.add_legend()
plt.show()

# Plot KDE for each gender with specified colors
sns.kdeplot(data.split_frac[data.gender == 'M'], label='men', fill=True, color='blue')
sns.kdeplot(data.split_frac[data.gender == 'W'], label='women', fill=True, color='green')
plt.show()

# Plot violin plot with updated syntax
sns.violinplot(x="gender", y="split_frac", data=data, hue="gender", palette={"M": "lightblue", "W": "lightpink"}, legend=False)
plt.show()

data['age_dec'] = data.age.map(lambda age: 10 * (age // 10))
print(data.head())

with sns.axes_style('white'):
    sns.violinplot(x="age_dec", y="split_frac", hue="gender", data=data,
                   inner="quartile", palette={"M": "lightblue", "W": "lightpink"})
plt.show()

# Create a FacetGrid to plot separately for each gender
g = sns.FacetGrid(data, col='gender', aspect=1.2, height=4)

# Map regplot to each subplot in the FacetGrid
g.map(sns.regplot, 'final_sec', 'split_frac', scatter_kws={'color': 'c'}, line_kws={'color': 'k'})

# Add horizontal lines at y=0.1 for each facet
for ax in g.axes.flat:
    ax.axhline(y=0.1, color="k", linestyle=":")
plt.show()



# # SLIDES NOTES:
from matplotlib.pyplot import boxplot

print("SLIDES NOTES")
# BAR CHART WITH SEABORN
print("\n\nBAR CHART WITH SEABORN\n")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

categories = ['A', 'B', 'C', 'D']
values = [5, 7, 3, 8]

sns.barplot(x=categories, y=values)

plt.title("Bar Chart with Seaborn")
plt.xlabel("Category")
plt.ylabel("Values")

plt.show()

rng = np.random.RandomState(30)
data_values = rng.randint(0, 100, 1000)
data_values2 = rng.randint(0, 100, 1000)
# print(data_values)

# histogram
print("\nHistogram With SeaBorn")
sns.histplot(data_values, bins=50, fill=True, kde=True)
sns.histplot(data_values2, bins=50, fill=True, kde=True)
plt.xlabel("x_v")
plt.show()

print("\nLine Chart With SeaBorn")
x = [1, 2, 3, 4, 5]
y = [10, 11, 12, 13, 14]

sns.lineplot(x=x, y=y, marker='o', label='line 1')

plt.title("Basic Line Plot with Seaborn")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.show()

print("\nPie Chart With SeaBorn")
x = [1, 2, 3, 4, 5]
y = [10, 11, 12, 13, 14]

print("Pie chart with seaborn")
# type 1
data = {'labels': ['A', 'B', 'C', 'D'],
        'values': [20, 35, 25, 20]}
df = pd.DataFrame(data)
sns.set_style("whitegrid")
plt.figure(figsize=(6,6))
plt.pie(df['values'], labels=df['labels'], autopct='%1.1f%%')
plt.title('Type 1: My Pie Chart')
plt.show()

# Type 2
# Create pie chart
data = [10, 20, 30, 40]
labels = ['A', 'B', 'C', 'D']
sns.set_style("darkgrid")
plt.pie(data, labels=labels)

# Add title
plt.title("Type 2: Distribution of Data")

# Show plot
plt.show()

# Type 3
# Create pie chart
data = [10, 20, 30, 40]
labels = ['A', 'B', 'C', 'D']
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
sns.set_style("darkgrid")
plt.pie(data, labels=labels, colors=colors)

# Add title
plt.title("Type 3: Distribution of Data")

# Show plot
plt.show()

# Type 4
# Create pie chart
data = [10, 20, 30, 40]
labels = ['A', 'B', 'C', 'D']
explode = (0, 0.1, 0, 0)
sns.set_style("darkgrid")
plt.pie(data, labels=labels, explode=explode)

# Add title
plt.title("Type 4: Distribution of Data")

# Show plot
plt.show()

# Type 5
sales_data = [20, 30, 15, 10, 25]
products = ['Hats', 'T-shirts', 'Pants', 'Jackets', 'Shoes']

# Explode Shoes slice
explode = [0, 0, 0, 0, 0.1]

# Create pie chart
plt.pie(sales_data, labels=products, explode=explode)

plt.title("Type 5: Pie Chart")

# Show plot
plt.show()


print("\n\nScatterplots with seaborn")
tips = sns.load_dataset("tips")
print(tips.head())

sns.scatterplot(tips, x=tips['total_bill'], y=tips['tip'])
plt.show()

sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time")
plt.show()

sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", style='time')
plt.show()


print("\n\n\nBoxplot/box plot with seaborn")
# example 1
sns.set_theme(style="ticks", palette="pastel")

# Load the example tips dataset
tips = sns.load_dataset("tips")

# Draw a nested boxplot to show bills by day and time
sns.boxplot(x="day", y="total_bill",
            hue="smoker", palette=["m", "g"],
            data=tips)
sns.despine(offset=10, trim=True)
plt.show()

# example 2
data = {
        'Category' : ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'Values' : [7, 8, 9, 12, 10, 14, 6, 5, 8]
}

df = pd.DataFrame(data)
# palette alone is deprecated so better use hue=x_axis and legend=False for same effect as palette alone before
sns.boxplot(x='Category', y='Values', data=df, hue='Category', legend=False, palette='Set3')

plt.title("Boxplot with Seaborn")
plt.xlabel("x_axis")
plt.ylabel("y_axis")

plt.show()