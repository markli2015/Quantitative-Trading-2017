########################################
#
#    Quantitative Trading in Python
#
########################################




'Python Crush Course'
------------------------

import math 

math.sqrt(10) # check other operation

range(5) # list generator
range(1,5,1) # start, end, step size


'choose a method when func_object. type tab to check'

'use """ """ in func specification, active in func(|), shift+tab, will see doc string'
'Also used to check func'

list(map(func,seq)) # transform
list(filter(func,seq)) # 


# remove from my list
A = range(0,5)
A.pop(0) # remove the first value
A.pop(0) # remove the fisrt value of the first result
A 
'[2,3,4]'



# --------- Numpy

'automatically choose float format in case loss any data'

np.zeros(4) # i dimension
np.zeros((5,5)) # 5 X 5 2 dimensions
np.zeros(((5,5,5))) # 5 X 5 X 5 3 dimensions

arr = np.array([.......])
arr.dtype
"dtype('int32')"

np.nan # generate NaN value

arr.max
'max value..'
arr.min

arr.argmax()
'index of max'
arr.argmin()


# as long as ran the first, run second multiple times gets the same
np.random.seed(101)
np.random.rand(1)

# create 2D matrix (row,col,....) of random numbers
np.random.randn(1000,2)


arr/0 # will staill run and return warining
'[Inf,Inf,....]'

'A lot of operations'
np.exp(arr)   np.sin(arr)



arr[0:3] = 100 # broacast: change all arnge to 100s
arr

# new assignment is just a pointer, change to new assignment impacts original array (save memeory)
arr = np.array([0,1,2,3,4,5,6,7,8,9])
arr_sub = arr[0:6]
arr_sub[:] = 10
arr
'array([10,10,10,10,10,10,6,7,8,9])'

# Or
arr_sub = arr[0:6].copy() # new object, not copy


# quick filter
arr = np.array([0,1,2,3,4,5,6,7,8,9])
arr[arr>4]
'array([5,6,7,8,9])'


# indice matrix
mat = np.array([[1,1,1],[2,2,2],[3,3,3]])
mat[1,2] # 2 row, 3 column
mat[1][2] # same
mat[:2,:1] # can slice mat






# ------- Pandas

# Series
'Similar to numpy arrays, except we can give then a named or datetime index,' 
'instead just numeric values'

import pandas as pd
S = pd.Series([1,2,3,4],index=['a','b','c','d'])

S['a']
1

#not usual way of doing
pd.Series([sum,print,len]) # can store func

'operation, will do with same index pair'
S1 + S2



# Data Frame
df['col1']    df[['col','col2','col3']] # subset columns

df.drop('col1',axis=1) # not affect df
df.drop('col1',axis=1,inplace=True) # will affect


df > 0 # returen F/T of df for each value
df[df>0] # use for filter, not match show ' NaN'

df[df['col1']>0] # select all row that 'col1' > 0, better than above

True and False # 'and' works
[True,False,True] & [False,True,False] # 'and' works
df[(df['col1']>1) & (df['col2']<0)]
'also' |


df.reset_index() # re order 1 - .. index, if original index 'label', new column as 'index'
df.set_index('col1') # use col1 as new index of df



# Multi-index of Data Frame
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside)) # list of tuple with 2 value
hier_index = pd.MultiIndex.from_tuples(hier_index) # create index object
df = pd.DataFrame(randn(6,2),hier_index,['A','B']) # data value, index, column names
df.index.names = ['outside','inside'] # give each index catwegory a name

df.loc['G1'].loc[1]['B'] # indice a value, from outside to inside
df.xs(1,level='inside') # indice to get all value under a index cat value


# Create dataframe from dict
d = {'A':[1,2,3],'B':[3,2,1]}
df = pd.DataFrame(d) # 2 column, A and B


# drop missing
df.dropna() # by row
df.dropna(axis=1) # by column
df.dropna(thresh=2) # keep at least 2 Non-NaN values

df.fillna(value=30) # fill all NaN with 30
df['col1'].fillna(value=df['A'].mean()) # fill mean to missing value



# Group By
df.groupby('col1').sum().loc['index2'] # aggreagted index2 value over all numeric cols
df.groupby('col1').describe() # give a lot of aggregated measures (vertical)
df.groupby('col1').describe().transpose() # give a lot of aggregated measures (horizontal)



# Merge Join, concate
pd.concat([df1,df2,df3],axis=1) # by column, index names need to match
pd.concat([df1,df2,df3],axis=0) # by row, column names need to match

pd.merge(df1,df2, on=['key1'], how='outer') # join on columns
df1.join(df2, how='inner') # join on index




# Common operations
df['col1'].unique() # all unique value in aaray
df['col1'].nunique() # count all unique value 
df['col1'].value_counts() # how many time each unique value occurs
df['col1'].apply(func) # every element of that series
df['col1'].apply(lambda x: x*2) # every element of that series
df.columns # column names in list
df.index # index names in list
df.sort_values('col2') # sort df by col2
df.isnull() # blooun of F/T in same size df




# data input output

pwd # current wd

pd.read_csv('xxxx.csv') # read csv
pd.read_ 'tab and see all method to read'

df.to_csv('xxxxx',index=False) # no index, or new index when read in

pd.read_excel('xxxx.xlsx',sheetname='Sheet1') # read xlsx
df.to_excel('xxxx.xlsx',sheet_name='NewSheet') # write xlsx

data = pd.read_html('https://www.xxxx/xxxx/xxx/xxx.html') # find any table element in html code
'return a list of df of all tables '
data[0].head() # first data frame













# ----------- Matplotlib
import matplotlib.pyplot as plt
%matplotlib inline # allow you to plot within JuP notebook

# >> Functional method
plt.plot(x,y)
plt.show() # needed when in Jupiter notebook

plt.plot(x,y,'r') # use red color line
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.title('title')

# multiple plot
plt.subplot(1,2,1) # 1 row, 2 plot, No1
plt.plot(x,y,'r')
plt.subplot(1,2,2) # 1 row, 2 plot, No2
plt.plot(y,x,'b')



# >> Object-oriented method
fig = plt.figure() # canvas object
axes = fig.add_axes([0.1,0.1,0.8,0.8])

axes.plot(x,y)
axes.set_xlabel('xlabel')
axes.set_ylabel('ylabel')
axes.set_title('Set Title')

# Nested plot
fig = plt.figure() # canvas object
axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
axes2 = fig.add_axes([0.5,0.5,0.8,0.8])

axes1.plot(x,y)
axes1.set_title('LARGER PLOT')

axes2.plot(y,x)
axes2.set_title('SMALLER PLOT')


# Multiple plot
fig,axes = plt.subplots(nrow=1,ncol=2)
plt.tight_layout() # adjust overlapping index (recommend always inclued by the end)

axes[0].plot(x,y)
axes[0].set_title('LARGER PLOT1')

axes[1].plot(y,x)
axes[1].set_title('LARGER PLOT2')


# >>> Figure Size and DPI
fig,axes = plt.subplots(nrows=1,ncols=2)
axes
'array of matplotlib.axes objects'

# Signle plot
fig = plt.figure(figsize=(3,2),dpi=100) # dot-per-inch (usually choose default) configure
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y) 

# multi-plot
fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(8,2))
axes[0].plot(x,y)
axes[1].plot(y,x)
plt.tight_layout()

# Save a fig
fig.savefig('my_picture.jpg')
fig.savefig('my_picture.png',dpi=200) # change pic setting

# Add legend (two lines in one plot)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,x**2,label='X Squared') # line 1
ax.plot(x,x**3,label='X Cubed') # line 2
ax.legend() 
'or'
ax.legend(loc=0) # 0,1,2 - check position
'or'
ax.legend(loc=(0.1,0.1)) # coordinates position



# >>>> Plot Appearance
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y,color='red') # red, green, blue, orange
ax.plot(x,y,color='#FF8C00') # rRGB Hex code
ax.plot(x,y,linewidth=1) # 1 width [lw]
ax.plot(x,y,alpha=0.5) # transparency
ax.plot(x,y,linestyle=':') # -- :,.,step,... [ls] 
ax.plot(x,y,marker='o', # +,1,*,o,... data points
            markersize=1, # dot size
            markerfacecolor='yellow', # dot color
            markeredgewidth=3, # dot circle edge width
            markeredgecolor='green') # dot circle edge color 

ax.set_xlim([0,1]) # x axis
ax.set_ylim([0,1]) # y axis



# other plot types
plt.scatter(x,y)
plt.hist(sample(range(1,1000),100))
plt.boxplot([np.tandom.normal(0,std,100) for std in range(1,4)])
'matplotlib plot types...'















# ----------- Pandas Visualization + seanorn

import numpy as np
import pandas as pd
import seaborn as sns # better plotting style (automatically apply)
%matplotlib inline

df1 = pd.read_csv('df1',index_col=0) # use first column as index (good for time series)

# Plot with dataframe
df1['col1'].hist(bins=30) # create hist gram in matplotlib plot
'Or'
df1['col1'].plot(kind='hist',bins=30) # create hist gram in matplotlib plot
'or'
df1['col1'].plot.hist(bins=30) # create hist gram in matplotlib plot


df1.plot.area() # area plot for all columns (numeric columns)
'or'
df1.plot.area(alpha=0.4) # same plot but more transparency

df1.plot.bar() # take index as categorical (horizontal) and bars for each columns across
'or'
df1.plot.bar(stacked=True) # same above by stacked into one bar for each index

df1.plot.line(x=df1.index,y='col2') # line plot for column col2 and index as x-axis
'or'
df1.plot.line(x=df1.index,y='col2',figsize=(12,3),lw=3) # line plot with widder canvas, thicker line

df1.plot.scatter(x='col1',y='col2',c='col3') # scatter plot of x,y and color coded on col3 value
'or'
df1.plot.scatter(x='col1',y='col2',c='col3',cmap='coolwarm') # same plot, red-blue color scale
'or'
df1.plot.scatter(x='col1',y='col2',s=fd1['col3']*100) # same plot, size (X100) coded by col3

df1.plot.box() # boxplot for each of the numeric column (column names as horizontal)

df1.plot.hexbin(x='col1',y='col2',gridsize=25) # like scatter plot but in hexigons 
'or'
df1.plot.hexbin(x='col1',y='col2',gridsize=25,cmap='coolwarm') # same above, red-blue

df1['col'].plot.density() # smoothed, kernal density plot of a series
'or'
df1.plot.density() # density plot for all numeric columns



# Visualizing time series -
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

mcdon = pd.read_csv('xxx.csv',index_col='Date',parse_dates=True) # read csv file, use 'Date' as index, parse_dates = time series

# value columns at different scales (call series separately)
mcdon['col1'].plot(figsize=(12,4)) # long canvas
mcdon['col2'].plot(figsize=(12,4)) # long canvas

# interested in sepacific date period
mcdon['col2'].plot(xlim=['2007-02-01','2009-01-01'],ylim=(20,50)) # x,y limts reset to get clear view difference




# ----- change date X-axis with custom plot
# Extract information
idx = mcdon.loc['2007-01-01':'2007-05-01'].index
stock = mcdon.loc['2007-01-01':'2007-05-01']['col1']

# Plotting
fig,ax = plt.subplots()
ax.plot_date(idx,stock,'-')

# Major
ax.xaxis.set_major_locator(dates.MonthLocator()) # identify month
ax.xaxis.set_major_formatter(dates.DateFormatter('\n\n\n\n%b-%Y')) # Month format (can change)

# Minor
ax.xaxis.set_minor_locator(dates.WeekdayLocator(byweekday=0)) # 0 sun, 1 mon,... identify weekday
ax.xaxis.set_minor_formatter(dates.DataFormatter('%d')) # weekday

# adjust plotting
fig.autofmt_xdate() # automatically adjust allignment
plt.tight_layout()



# ----- ONLY Jupitor notebook interactive plot
'Restart the kernal - on top banner'

'change' %matplotlib inline 'to' %matplotlib notebook

df1.plot.line(x=df1.index,y='col2') 

'Interactive features on plot bottom'























'Data Source'
------------------------

-1 'Pandas Data Reader' # allows connect to various API to pull data in as DF

import pandas_datareader.data as web
import datetime

start = datetime.datetime(2015,1,1)
end = datetime.datetime(2017,1,1)

facebook = web.DataReader('FB','google',start,end) # company ticker, source api, start date, end date
facebook.head()

# Get Options data frame
from pandas_datareader.data import Options
fb_options = Options('FB','google')

option_df = fb_options.get_options_data(expiry=fb_options.expiry_dates[0])
option_df.head()






-2 'Quandl' # offer robust Python API (also in R) free and paid data

'quandl.com' # more data - alternative data + core financial data
# Need to also register account 

# test free version - 50 request / day

- 'core financial data'
- 'free check box'
- 'select wiki prices'
# should see 3000 US company stock pricess

- 'On the right side, select Python table'
- 'select date range on top'
- 'Copy sample REQUEST code in Python'

import quandl
mydata = quandl.get('EIA/PET_RWTC_D') # oil price 
mydata.head()

mydata = quandl.get('WIKI/AAPL') # apple stock
'or'
mydata = quandl.get('WIKI/AAPL.1') # apple stock, only first column

# find the code 'EIA/PET...' in web page "data" explore
















''
------------------------


