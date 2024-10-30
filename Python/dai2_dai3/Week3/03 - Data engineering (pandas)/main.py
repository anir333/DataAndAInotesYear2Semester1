# import numpy as np
# import pandas as pd
#
# print(np.__version__)
# print(pd.__version__)
#
# series = pd.Series([1, 2, 3, 4, 5]) # one-dimensional array of indexed data
# print(series)
# print(series.values)
# print(series.index)
# print(series[1:3])
# data = pd.Series([0.25, 0.5, 0.75, 1.0],
#                  index=['a', 'b', 'c', 'd']) # using strings as indexes
# print(data)
# print(data['b'])
# data = pd.Series([0.25, 0.5, 0.75, 1.0],
#                  index=[2, 5, 3, 7])
# print(data)
# print(data[5])
#
# population_dict = {'California': 38332521,
#                    'Texas': 26448193,
#                    'New York': 19651127,
#                    'Florida': 19552860,
#                    'Illinois': 12882135}
# population = pd.Series(population_dict) # converting python dictionary to pandas series
# print(population)
# print(population['California':'Illinois'])
#
# # ``data`` can be a scalar, which is repeated to fill the specified index:
# pd.Series(5, index=[100, 200, 300])
#
#
# # ``data`` can be a dictionary, in which ``index`` defaults to the sorted dictionary keys:
# pd.Series({2:'a', 1:'b', 3:'c'})
# pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2]) # index will be 3, 2
#
#
# ## DataFrame:
# # we'll use a series first:
# area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
#              'Florida': 170312, 'Illinois': 149995}
# area = pd.Series(area_dict)
# print(area_dict) # series
# print()
#
# states = pd.DataFrame({'population': population,
#                        'area': area})
# print("states = \n", states) # population and area are columns with the same indexes, but different data
# print(states.index)
# print(states.columns)
# print()
# print(states['area'])
# print(states['area']['California'])
# print()
#
# print(pd.DataFrame(population, columns=['population']))
#
# data = [{'a': i, 'b': i*2} for i in range(3)]
# print(data)
#
# # if keys in dictionary are missing pandas will fill them with NaN (not a number)
# print()
# print(pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}]))
#
# print()
# print(pd.DataFrame(np.random.rand(3, 2),
#              columns=['foo', 'bar'],
#              index=['a', 'b', 'c']))
#
# print()
# A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
# print(A)
#
# ## Pandas Index Object
# ind = pd.Index([2, 3, 5, 7, 11])
# print()
# print(ind)
# print(ind[1])
# print(ind[::2])
# print(ind.size, ind.shape, ind.ndim, ind.dtype) # values in indexes cannot be changed once declared (final/constant immutable values)
# #indexes have similar conventions used by Set in Python:
# indA = pd.Index([1, 3, 5, 7, 9])
# indB = pd.Index([2, 3, 5, 7, 11])
# print(indA & indB) # intersection (will return an index array of only values that are in both index arrays
# print(indA | indB) # union (will return a union index array ==> all values in both without repetitions)
# print(indA ^ indB) # Symmetric difference (will return an index object array of the values that are different in each other)
# print()
#
#
#
#
#
# df_d = pd.DataFrame(np.arange(12).reshape(3, 4), index=['a', 'b', 'c'], columns=['x', 'y', 'z', 'w'])
# print(df_d)
# print()
#
# # df_d['x']['a'] = 3
# print(df_d.loc['a', ['x', 'y']])
# print(df_d.at['a', 'z'])
# print(df_d.iloc[0, 2])
#
# # with loc you specify the actual index of the row you want to select, with iloc you use the python implicit convention index to select the rows you want
# """more dataframes methods
# Basic Access
#
# df.head(n): Returns the first n rows of the DataFrame.
# df.tail(n): Returns the last n rows of the DataFrame.
# df.shape: Returns the dimensions of the DataFrame.
# df.columns: Returns the column labels of the DataFrame.
# df.index: Returns the index (row labels) of the DataFrame.
# Selection by Label
#
# df.loc[row_label, column_label]: Access a group of rows and columns by labels.
# df.at[row_label, column_label]: Access a single value by label.
# Selection by Position
#
# df.iloc[row_index, column_index]: Access a group of rows and columns by integer positions.
# df.iat[row_index, column_index]: Access a single value by position.
# Boolean Indexing
#
# df[df['column'] > value]: Select rows based on column values.
# Accessing Column(s)
#
# df['column_name']: Access a single column.
# df[['column1', 'column2']]: Access multiple columns.
# Modifying Data
# Adding or Modifying Columns
#
# df['new_column'] = value: Add or modify a column.
# df.assign(new_col1=value1, new_col2=value2): Add new columns.
# Adding Rows
#
# df.append(other_df): Append rows of another DataFrame.
# df.loc[new_index] = values: Add a new row.
# Removing Columns or Rows
#
# df.drop('column_name', axis=1): Drop a column.
# df.drop(index): Drop a row.
# df.drop(columns=['column1', 'column2']): Drop multiple columns.
# df.drop([index1, index2]): Drop multiple rows.
# Updating Values
#
# df.at[row_label, column_label] = new_value: Update a single value by label.
# df.iat[row_index, column_index] = new_value: Update a single value by position.
# df.loc[row_condition, 'column_name'] = new_value: Update values based on a condition.
# Advanced Modification
# Replacing Values
#
# df.replace(to_replace, value): Replace values throughout the DataFrame.
# df.replace({col1: {old_value1: new_value1}, col2: {old_value2: new_value2}}): Replace values in specific columns.
# Renaming Columns and Index
#
# df.rename(columns={'old_name': 'new_name'}): Rename columns.
# df.rename(index={'old_index': 'new_index'}): Rename rows.
# Setting Index
#
# df.set_index('column_name'): Set a column as the index.
# df.reset_index(): Reset the index to the default integer index.
# Sorting
#
# df.sort_values(by='column_name'): Sort by values in a column.
# df.sort_index(): Sort by index.
# Applying Functions
#
# df.apply(func): Apply a function along an axis of the DataFrame.
# df.applymap(func): Apply a function element-wise.
# df['column_name'].map(func): Apply a function element-wise on a column.
# Combining DataFrames
# Concatenation
#
# pd.concat([df1, df2], axis=0): Concatenate along rows.
# pd.concat([df1, df2], axis=1): Concatenate along columns.
# Merging and Joining
#
# pd.merge(df1, df2, on='key'): Merge DataFrames on a key.
# df1.join(df2, on='key'): Join DataFrames on a key.
# Aggregation and Grouping
# Grouping
#
# df.groupby('column'): Group by a column.
# df.groupby(['col1', 'col2']): Group by multiple columns.
# Aggregation
#
# df.agg({'column1': 'sum', 'column2': 'mean'}): Aggregate using functions.
# df.groupby('column').agg({'col1': 'sum', 'col2': 'mean'}): Group and aggregate.
# Pivot Tables
#
# df.pivot(index='index_col', columns='column_col', values='value_col'): Create a pivot table.
# df.pivot_table(index='index_col', columns='column_col', values='value_col', aggfunc='mean'): Create a pivot table with aggregation.
# Reshaping
# Stacking and Unstacking
#
# df.stack(): Pivot columns to rows.
# df.unstack(): Pivot rows to columns.
# Melt
#
# df.melt(id_vars=['id_col'], value_vars=['val_col1', 'val_col2']): Unpivot a DataFrame from wide to long format.
#
#
# """
#
# # fancy series indexing and functionalities
# data = pd.Series([0.25, 0.5, 0.75, 1.0],
#                  index=['a', 'b', 'c', 'd'])
# print()
# print("data = \n", data)
# print()
# print(data['b'])
# print('a' in data) # True
# print(data.keys())
# print(list(data.items()))
# data['e'] = 1.25
# print(data)
# print(data['a':'c'])
# print(data[0:2])
# # masking
# print(data[(data > 0.3) & (data < 0.8)])
# print(data[['a', 'e']])
# """
# data.loc[1]
# data.loc[1:3] # specifically indexes 1 and 3 only ==> s you are calling the existing indexes
# data.iloc[1]# using python style indexing with iloc
# data.iloc[1:3] # fropm index 1 to 2 (cause we exclude 3 like in python)
#
# """
# print()
#
#
#
# # fancy dataframes (there are methods before this and before fancy series about dframes fancy methods
# area = pd.Series({'California': 423967, 'Texas': 695662,
#                   'New York': 141297, 'Florida': 170312,
#                   'Illinois': 149995})
# pop = pd.Series({'California': 38332521, 'Texas': 26448193,
#                  'New York': 19651127, 'Florida': 19552860,
#                  'Illinois': 12882135})
# data = pd.DataFrame({'area':area, 'pop':pop})
# print(data)
# print(data.area is data['area']) # true
# print(data.pop is data['pop']) # false because method pop exists so it's conflicting with it
#
# #adding a new column + using already existing data to do so:
# print(0)
# print("data as a DataFrame: ")
# data = pd.DataFrame(data)
# data['density'] = data['pop'] / data['area']
# print(data)
# print(data.values)
# print(data.T) # the dataframe will swap rows and columns
# print(data.iloc[:3, :2]) #allows us to access specific data just like in standard python convention (or like numpy arrays) ==> the order is rows and columns
# print(data.loc[:'Illinois', :'pop']) # order is rows, columns, it includes them, not like in python convention
#
# #data.ix allows a mix of python convention and pandas convention, but with the order being the pandas convention (column, row)
# # DISCLAIMER: DOESN'T WORK ON MOST RECENT VERSIONS OF PANDAS
# # print(data.ix[:3, :'pop']) # using square brackets
#
# ## Fancy dataframe indexing:
# print(data.loc[data.density > 100, ['pop', 'density']])
# # from columns pop and density, shows the rows that comply with the condition
#
# #modidfy values
# data.iloc[0, 2] = 90
# print(data)
#
# ## More indexing conventions:
# print(data['Florida':'Illinois']) # here we are accessing them by row
# print()
# print(data[1:3]) # here we are also referring to rows (so same as the previous one on how it works)
# print()
# print(data[data['density'] > 100]) # selects only the rows that comply with the condition
# # print(data[data.density] > 100) # ==> this also works but better use the previous one to avoid errors with confusion with already existing methods that think are being called here
#
#
# print("----------------------------------------")
#
# # using random as in notebooks
# rng = np.random.RandomState(42)
# ser = pd.Series(rng.randint(0, 10, 4)) # give me a series (dictionary of indexes and values) of 4 integer numbers from 0 to 10
# df = pd.DataFrame(rng.randint(0, 10, (3, 4)),
#                   columns=['A', 'B', 'C', 'D']) # give me a dataframe (multidimensional array/dictionary) with integers form 0 10, give me 3 rows and 4 columns with column name being A, B, C, D
# # you can operate normally in series and dataframes justlike in numpy arrays
# print(df * 5)
#
# # when trying to operate with two different series/dataframes, missing indexes will be given the value NaN on the result of the operation:
# A = pd.Series([2, 4, 6], index=[0, 1, 2])
# B = pd.Series([1, 3, 5], index=[1, 2, 3])
# print(A + B)
# # if you don't want to have the NaN value, you can use fill_value=0
# A.add(B, fill_value=0) # this means that it will integrate the missing values that are in B but not in A into A, it won't initialise them to 0, it will keep their content they had in B, just transferring them to A because its basically performing addition from values of B to A, so for values that are empty in A, it will instead perform an addition of 0 + B value instead of Nan + B value
#
# # pandas aligns indexes on operations, so when you do Df1 + Df2 or Serie_1 + Serie_2, it won't sum them by 'physical' position, but by actual index matching
# A = pd.DataFrame(rng.randint(0, 20, (2, 2)),
#                  columns=list('AB'))
# print("A:\n", A)
# B = pd.DataFrame(rng.randint(0, 10, (3, 3)),
#                  columns=list('BAC'))
# print("B:\n", B)
# print(A + B)
#
#
# #  we can also add them and fill the empty values with other information, example, we do the mean of all values in A and then add that mean to each value missing in A that's in B, and put it in A:
# fill = A.stack().mean()
# print(A.add(B, fill_value=fill)) # A will contain the sum of all values in the same position in B, and for those missing it will calculate the mean of A (before any calculation changes it) and then add that to the values of B that are not in A and put them in A
#
# """
# The following table lists Python operators and their equivalent Pandas object methods:
#
# | Python Operator | Pandas Method(s)                      |
# |-----------------|---------------------------------------|
# | ``+``           | ``Serie/DataFrame/Object.add()``      |
# | ``-``           | ``sub()``, ``subtract()``             |
# | ``*``           | ``mul()``, ``multiply()``             |
# | ``/``           | ``truediv()``, ``div()``, ``divide()``|
# | ``//``          | ``floordiv()``                        |
# | ``%``           | ``mod()``                             |
# | ``**``          | ``pow()``                             |
# """
#
# # Broadcasting works the same in pandas as in numpy arrays:
# print("------------------------")
# A = rng.randint(10, size=(3, 4))
# print(A)
# print(A - A[0]) # normal python array
#
# df = pd.DataFrame(A, columns=list('QRST'))
# print(df)
# print(df - df.iloc[0]) #dataframe broadcasting the first row the same way
#
# # same operation (broadcasting) but column wise
# print(df.subtract(df['R'], axis=0))
#
# # datafram/serues automatically align indexes between the two elements on operations:
# print("halfrow")
# halfrow = df.iloc[0, ::2]
# print(halfrow.__class__) # a Series because we only selected on row so one dimension
# print(halfrow)
# print(df - halfrow)
# print()
#
# hf = pd.DataFrame(df.loc[:, ['R', 'T']].values, columns=list('RT'))
# hz = pd.DataFrame([0, 1, 2], columns=list('Z'))
# # print(hz.add(hf))
# # hf.add(pd.DataFrame([0, 1, 2], columns=list('Z')))
# print(hf)
#
# print("--------------------------------")
# # numpy with None/NaN values, cannot handle None of python, will convert the numpy objects to python objects
# vals2 = np.array([1, np.nan, 3, 4])
# # vals2.dtype is equal to float64 cause nan is a floating point number in IEEE convention
# # won't throw errors when implementing aggregate functions like sum, min... will simply return nan
# # to avoid nan numbers you cna use:
# np.nansum(vals2)
# np.nanmin(vals2) # and more, this ony performs the operation on non nan values
#
#
# # pandas with None/NaN it can handle them both, converts python None to numpy NaN (np.nan) which is a floating point number per convention
# pd.Series([1, np.nan, 2, None]) # the None will be converted to np.nan (NaN)
# data = pd.Series([1, np.nan, 'hello', None])
# print(pd.isnull(np.nan)) # True
# print(pd.notnull(np.nan))# False
# print(data.isnull())
# print(data[data.notnull()])
# print("---------------------------")
# df = pd.DataFrame([[1,      np.nan, 2],
#                    [2,      3,      5],
#                    [np.nan, 4,      6]])
# print(df.dropna())
# print((df - halfrow).dropna())# in this case all rows will be removed and the dataframe will be empty because df.dropna() removes all rows that have at least one NaN value
# print(df.fillna(130))
# print(df.dropna(axis='rows')) # default
# df[3] = np.nan
# print("new df with full NaN column:")
# print(df)
# print(df.dropna(axis='columns'))
# print(df.dropna(axis='columns', how='all')) # will only drop columns if they are all NaN values
# print(df.dropna(axis='rows', thresh=3)) # only will drop rows with a NaN value which have less than 3 non-null values, so if the row has at least three non-null values (Example three integers), then it will not drop the row even if there is another value that is NaN
#
# print("----------------FIlling nulls---------------")
# data = pd.Series([1, np.nan, 2, None, 3, np.nan, None, None, None], index=list('abcde1234'))
# # DataFrame/Serie/.fillna(value to fill)
# print(data.fillna(0))
# print(data.ffill()) #Forward fill 1, 2, 3, 4, ...
# print(data.bfill())
# print()
# print(df)
# print(df.ffill(axis=1)) # in dataframes you can specify the axis since they're multidimensional
# print(df.ffill(axis=0))
#
#
# print("--------------------------Joins and Merge--------------------")
# df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
#                     'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
# df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
#                     'hire_date': [2004, 2008, 2012, 2014]})
# print("DF1")
# print(df1)
# print("DF2")
# print(df2)
# print("DF3 (One-to-one) pd.merge(df1, df2)")
# df3 = pd.merge(df1, df2)
# print(df3)
# print("DF4")
# df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR', 'new'], 'supervisor': ['Carly', 'Guido', 'Steve', 'nneeww']})
# print(df4)
# print('DF5 (Many-to-one) df3.merged(df4)')
# df5 = df3.merge(df4)
# print(df5)
# print('DF6')
# df6 = pd.DataFrame({'group': ['Accounting', 'Accounting','Engineering', 'Engineering', 'HR', 'HR'], 'skills': ['math', 'spreadsheets', 'coding', 'linux', 'spreadsheets', 'organization']})
#
# print(df6)
# print('DF7 (Many-to-any) df1.merge(df6)')
# df7 = df1.merge(df6)
# print(df7)
#
# """
# When we joins two DataFrames they are joined by the common column
# One-to-one join/merge ==> when both columns on each table do not have repetitions of the same value
# Many-to-one join/merge/ ==> when one of the columns on both tables has many repetitions of a single value
# Many-to-many join/merge ==> when both columns on both tables have multiple repetitions of a single value (the same elements on other columns might be doubled since in other columns being joined are different)
# """
# print()
# print(pd.merge(df1, df2, on='employee')) # to specify columns to use as key to join the tables
# print('DF3')
# df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
#                     'salary': [70000, 80000, 120000, 90000]})
# print(df3)
# # if for example the same column has diferent names on diferent tables, you can specify the column name of each table to use for the join:
# print(pd.merge(df1, df3, left_on="employee", right_on="name"))
# # mto rop the extra redundant column added due to this join:
# print(pd.merge(df1, df3, left_on="employee", right_on="name").drop('name', axis=1))
# # convert a column to an index o the table:
# df1a = df1.set_index('employee')
# df2a = df2.set_index('employee')
# print(df1a) # now the employee is the index of the table
# print(df2a) # now the employee is the index of the table
# # then you can merge them by index and not get double columns:
# print("\nJoining by index:\n") # merging on index merge in index merge on index merge with index merging with index
# print(pd.merge(df1a, df2a, left_index=True, right_index=True))
#
# # the df1.join(df2) directly joins using indexes
# print(df1a.join(df2a))
# # you can combine joins of indexes with columns:
# print(pd.merge(df1a, df3, left_index=True, right_on='name'))
#
# # when performing a join of two dataframes (tables), we do it through the common column, but what if there is only one value in each one that can corresponds to the value of the other table:
# df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
#                     'food': ['fish', 'beans', 'bread']},
#                    columns=['name', 'food'])
# df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
#                     'drink': ['wine', 'beer']},
#                    columns=['name', 'drink'])
# print(df6)
# print(df7)
# print("------Only Mary------")
# print(pd.merge(df6, df7)) # specifically this is an inner join
# # it can also be specified like so:
# print("\ninner join")
# print(pd.merge(df6, df7, how="inner"))
# #outer
# print("\nouter join") # fills empty values with NaN (np.nan)/None
# print(pd.merge(df6, df7, how="outer"))
# print("\nleft join") # joins what's on left table + what's common on both, but leaves out what's not common nor present in left table (so may leave out rows from right table basically)
# print(pd.merge(df6, df7, how="left"))
#
# print("\nright join") # joins what's common on both tables or/and what's on right table
# print(pd.merge(df6, df7, how="right"))
#
# #joins with two common columns in which the values aren't the same:
# print("\ntwo same column names on both tables:")
# df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
#                     'rank': [1, 2, 3, 4],
#                     'vals' : [1, 3, 3, 1]})
# df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
#                     'rank': [3, 1, 4, 2],
#                     'vals': [4, 1, 2, 2]})
# print(df8)
# print(df9)
# print("merge: (you need to specify the column to merge with cause there are multiple same name columns on the tables)") # it will merge them both but add x and y value to differentiate them
# print(pd.merge(df8, df9, on="name"))
# # if you don't want it to call the columns with x and y values you cna use a suffix:
# print(pd.merge(df8, df9, on="name", suffixes=("_L", "_R")))
# print()
#
# "merging states"
# pop = pd.read_csv('data/state-population.csv')
# areas = pd.read_csv('data/state-areas.csv')
# abbrevs = pd.read_csv('data/state-abbrevs.csv')
# # all = [pop, area, abbrevs]
# print(pop.head())
# # print()
# print(areas.head())
# print(abbrevs.head())
# print("\n merge many-to-many")
# merged = pd.merge(pop, abbrevs, how="outer", left_on="state/region", right_on="abbreviation")
# merged = merged.drop('abbreviation', axis=1)
# # print()
#
# print(merged.head())
# print("null? ", merged.isnull().any(), "\n")
# print(merged[merged['population'].isnull()].head())
# print()
# print(merged.loc[merged['state'].isnull(), 'state/region'].unique()) # remember loc is [row_label, column_label], so in this case it gives all the values of state/region column, that are null in each row of state column (so go through the state column, and for each that is null, give me the state/region) and then at the end we remove duplicates with unique
# print()
# merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
# # I want to go through the column 'state/region', and whenever the value is 'PR', then set the 'tate' value to 'Puerto Rico' (logic)  ## assignation here::
# merged.loc[merged['state/region'] == 'PR', 'state'] = 'Puerto Rico'
# # print(merged.loc[merged['state'].isnull(), 'state/region'].unique()) " now this will be null cause nop longer states are null
# print(merged.isnull().any())
#
# print(merged.head())
# print(areas.head())
# final = pd.merge(merged, areas, on="state", how="left")
# print(final.head())
# print(final.isnull().any()) # per column tells if any row is null ==> boolean return True/False True or False
#
# print(final.loc[final['area (sq. mi)'].isnull(), 'state'].unique()) # logic : for each ROW in area that is null, give me the STATE
# #another way to do it:
# print(final['state'][final['area (sq. mi)'].isnull()].unique()) # first you access the state column, and for each row in area where the value is null, give that state ==> order is: COLUMN, ROW_value
# print(final)
# final.dropna(inplace=True) # if you add inplace=True, you don't need to do the assignation 'final = final.dropna(inplace=True)' cause it's done automatically by pointer reference
# print(final.isnull().any())
#
# # execute a query on a DataFrame
# print()
# data2010 = final.query("year == 2010 & ages == 'total'")
# print(data2010.head())
# data2010.set_index('state', inplace=True)
# print(data2010.head())
# density = data2010['population'] / data2010['area (sq. mi)']
# density.sort_values(ascending=False, inplace=True)
# print(f"\n{density.head()}")
# print(f"\n{density.tail()}")
#
# # Chapter 8 - Aggregation and Grouping
# print("\n\n\n------Chapter 8 - Aggregation and Grouping------------\n")
#
# import seaborn as sns
# planets = sns.load_dataset('planets')
# print(planets.shape)
#
# rnd = np.random.RandomState(32)
# serie = pd.Series(rnd.rand(5))
# print(serie)
# print(f"\nsum: {serie.sum()}")
# print(f"mean: {serie.mean()}")
#
# df = pd.DataFrame({'A': rng.rand(5),
#                    'B': rng.rand(5)})
# print(df)
# print(f"\nDataFrame mean: \n{df.mean()}") # for each column by default
# print(f"\nDataFrame mean for each row: \n{df.mean(axis='columns')}") # axis columns means it will do it for each row wtf
# print(planets.dropna().describe())
# print("\ncount:\n", planets.count())
# print(planets.head(1)) # to get first row
# print(planets.tail(1)) # to get last row
# print("\n", df.median(), "\n")
# print("\n", df.min(), "\n")
# print("\n", df.max(), "\n")
# print("\n", df.std(), "\n")
# print("\n", df.var(), "\n")
# print("\n", df.prod(), "\n")
# print("\n", df.sum(), "\n")
# print()
#
# # the groupby returns an object GroupBy which can be used to calculate with aggregates (sum, median, and all others)
# df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
#                    'data': range(6)}, columns=['key', 'data'])
# print(df.groupby('key').sum())
#
# print()
# print(planets.groupby('method')['orbital_period'].median()) # logic => group by the column method, and give me the median value on the column orbital_period correspondent to each grouped value in method columns
# print(planets.head())
# print()
# for (method, group) in planets.groupby('method'):
#     print("{0:30s} shape={1}".format(method, group.shape))
#     # print(method)
#     # print()
#     # print(group)
#     # break
#     # print((f"{method} shape={group.shape}"))
#
# # print(planets.query("method == 'Astrometry'"))
#
# print("\n", planets.groupby('method')['year'].describe())
# print("\n", planets.groupby('method')['year'].describe()['25%'])
# described_grouped_by_column_method = planets.groupby('method')['year'].describe()
# print("\n\n")
# for column in described_grouped_by_column_method:
#     print(column)
#
#
# print("\n\n")
# rng = np.random.RandomState(0)
# df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'], 'data1': range(6),'data2': rng.randint(0, 10, 6)},columns = ['key', 'data1', 'data2'])
# print(df)
# print("\n", df.groupby('key').aggregate(['min', 'median', 'max']))
# print()
#
# print(df.groupby('key').aggregate({'data1':'min', 'data2':'max'}))
#
# print("\nfiltering:")
# # you can create your own filtering function/method
# def filter_func(x):
#     return x['data2'].std() > 4
# print("\n", df, "\n", df.groupby('key').std(), "\n\n", df.groupby('key').filter(filter_func))
#
# # Transformation (transform the data with a given operation)
# print(df.groupby('key').transform(lambda x : x - x.mean())) # this is basically doing the mean of each group and substracting it form each value in that group (take into account that columns are in different subgroups than rows, so values in data1 and data2 column will have different means even if they are in the same row because it groups by the value in the same row and same column not in different columns
#
# print() # apply method:
# # you can use the apply method to transform specific data, for example here we only transform the values in column data1 by dividing them by the sum of the grouped keys: (this is called normalizing values of data1 by data2) ==> DISCLAIMER: the apply method takes a DataFrame and requires you to return a Pandas object or scalar (what you do in the method is up to u)
# def norm_by_data2(x):
#     x['data1'] /= x['data2'].sum()
#     return x
# print(df.groupby('key').apply(norm_by_data2, include_groups=False))
# print()
# # you can pass a list of values to group by, so for example instead of using an existing column on the DataFrame to group by with, se can make our own groups:
# L = [0, 1, 0, 1, 2, 0]
# print(df.groupby(L).sum())
# print()
# df2 = df.set_index('key')
# print(df2) # df2 has key as an index, so what we can do is specify what each value on the index can be grouped with:
# mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
# print(df2.groupby(mapping).sum())
# # by default any python function can be used which inputs a value, by default it uses the index
# print("\n", df2.groupby(str.lower).mean())
# print()
# # you can combine the previous ways in one:
# print(df2.groupby([str.lower, mapping]).mean())
# print()
# # example of combining all the ways to groupby:
# decade = 10 * (planets['year'] // 10)
# print("\n\n\ndecade:\n", decade)
# decade = decade.astype(str) + 's'
# decade.name = 'decade'
# # planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)
#
# print()
# print(df.groupby('key')['data2'].idxmax()) # idxmax returns the index of the row in the df where it complies with the max value
#
print("\n\n\n---------Chapter 10 - Working with Strings-----------\n\n")
import pandas as pd
import numpy as np

data = ['peter', 'Paul', None, 'MARY', 'gUIDO']

names = pd.Series(data)
print(names.str.capitalize(), "\n")

monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])
"""
pandas string methods that are similar to python string methods, but you must call them through pandasObject.str.method():
|             |                  |                  |                  |
|-------------|------------------|------------------|------------------|
|``len()``    | ``lower()``      | ``translate()``  | ``islower()``    | 
|``ljust()``  | ``upper()``      | ``startswith()`` | ``isupper()``    | 
|``rjust()``  | ``find()``       | ``endswith()``   | ``isnumeric()``  | 
|``center()`` | ``rfind()``      | ``isalnum()``    | ``isdecimal()``  | 
|``zfill()``  | ``index()``      | ``isalpha()``    | ``split()``      | 
|``strip()``  | ``rindex()``     | ``isdigit()``    | ``rsplit()``     | 
|``rstrip()`` | ``capitalize()`` | ``isspace()``    | ``partition()``  | 
|``lstrip()`` |  ``swapcase()``  |  ``istitle()``   | ``rpartition()`` |

some of them return values, for example:
"""
print(monte.str.lower())
print(monte.str.len())
print(monte.str.startswith('T'))
print(monte.str.split())

"""
## Regular expressions
match()     ==> Call re.match() on each element, returning a boolean.
extract()   ==> Call re.match() on each element, returning matched groups as strings.
findall()   ==> Call re.findall() on each element 
replace()   ==> Replace occurrences of pattern with some other string
contains()  ==> Call re.search() on each element, returning a boolean
count()     ==> Count occurrences of pattern
split()     ==> Equivalent to str.split(), but accepts regexps
rsplit()    ==> Equivalent to str.rsplit(), but accepts regexps
"""
print("\n", monte.str.extract('([A-Za-z]+)', expand=False))
print(monte.str.findall(r'^[^AEIOU].*[^aeiou]$'))

"""
get()   ==> Index on each element
slice() ==> Slice each element
slice_replace() ==> Replace slice in each element with passed value
cat() ==> Concatenate strings
repeat() ==> Repeat values
normalize() ==> Return Unicode form of string
pad() ==> Add whitespace to left, right, or both sides of strings
wrap() ==> Split long strings into lines with length less than a given width
join() ==> Join strings in each element of the Series with passed separator
get_dummies() ==> extract dummy variables as a dataframe
"""

print("\n", monte.str.slice(0, 3)) # same as:
print(monte.str[0:3])

#get surnames
print("\n", monte.str.split().str.get(-1))
print(monte)

full_monte = pd.DataFrame({'name': monte,
                           'info': ['B|C|D', 'B|D', 'A|C',
                                    'B|D', 'B|C', 'B|C|D']})
print(full_monte)
print("\n INFO:\n", full_monte['info'])
print("\ninfo divided by | on truth table stating appearence\n")
print(full_monte['info'].str.get_dummies('|'))


# create a dataframe out of text on a json file (each line is a json object)
import json
data = []
with open('recipeitems-latest.json') as recipes:
    for line in recipes:
        data.append(json.loads(line))

df_recipes = pd.DataFrame(data)
print(df_recipes.shape)
print(df_recipes.iloc[0])
print()
print(df_recipes.ingredients.str.len().describe()) # analyze/analyse the length of each ingredient string and describe it
print()
print(df_recipes.name[np.argmax(df_recipes.ingredients.str.len())]) # logic : get the recipes name of the row which has the maximum value on the string ingredient length
print()
print(df_recipes.description.str.contains('[Bb]reakfast').sum()) # how many recipes have ad description including the word breakfast (uppercase or lowercase Bb)
print("\nhow many recipes have cinnamon as an ingredient:")
print(df_recipes.ingredients.str.contains("[Cc]innamon").sum())

df=pd.Series(['Leonardo DiCaprio',
'Meryl Streep',
'Denzel Washington',
'Scarlett Johansson',
'  Kevin De Bruyne',
'Natalie Portman',
'Leonel Messi',
' Tom Hanks ',
'Angelina Jolie',
'Christian Bale',
' Kevin Bacon '              ],name="famous")
df = df.str.strip() # to remove blank spaces
print(df)