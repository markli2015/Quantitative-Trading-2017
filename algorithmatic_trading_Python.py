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
















'Time Series'
------------------------


######### Datetime objects ############
# Use 'Pandas' special time series features

- 'DateTime index'
- 'Time Resampling'
- 'Time Shifts'
- 'Rolling and Expanding'

# Create datetime object
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from datetime import datetime

my_year = 2017
my_month = 1
my_day = 2
my_hour = 13
my_minute = 30
my_second = 15

# create a datetime object
my_date_time = datetime(my_year,my_month,my_day,my_hour,my_minute,my_second) # Unspecified slot default to 0
my_date_time.day # get day (other objects)

# Create datetime index
first_two = [datetime(2016,1,1),datetime(2016,1,2)] # python list of datetime objects
dt_ind = pd.DatetimeIndex(first_two) # convert to datetime index
# attach index to data
data = np.random.randon(2,2); cols = ['a','b']
df = pd.DataFrame(data,dt_ind,cols)

df.index.argmax()
'1' # latest day 0,1,... -> 1 (2 values)
df.index.max()
"Timestamp('2016-01-02 00:00:00')"
df.index.argmin(); df.index.min() # same above




########### Read in file as time series format ###############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# Method 1, read first, convert next
df = pd.read_csv('xxxx/xxxx.csv')
df.info() # summarize features in df

df['Date'] = pd.to_datetime(df['Date']) # convert one column in data into datetime
                                        # can add 'format=strings' to specify datetime
                                        # like, '%d/%m/%Y'
df['Date'] = df['Date'].apply(pd.to_datetime) # same above
df.info() # check 'datetime' object
df.set_index('Date',inplace=True) # Use datetime as index

# Method 2, read and convert same time
df = pd.read_csv('xxx/xxxx.csv',index_col='Date',parse_dates=True) # not controlable as above, but if you 
                                                                   # sure about date format, saving time
df.info()




############# Resampling ################
df.resample(rule='A').mean() # rules -- set of all possible time series offset strings
'Yearly mean'
df.resample(rule='BQ').mean() # rules -- set of all possible time series offset strings
'Business Quaterly mean'
df.resample(rule='A').max()
'Yearly max'

df.resample('A').apply(custom_func) # Apply selfdefined func



############# Visualization #############
df['Close'].resample('A').mean().plot(kind='bar') # Yearly end bar plot
                                                  # Using pandas visualization on df




########### Time shift ##############
'some model requires shift time forward or backward for modeling'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('xxx/xxxx.csv',index_col='Date',parse_dates=True) 
df.shift(periods=1) # forward, all shift one index forward, original first = NaN, original last = removed
df.shift(periods=-1) # backward, all shift one index backward, original first = removed, original last = NaN

df.tshift(freq='M').head() # shift all index to match the time index as end of each month, no data loss, just index changed




############ Pandas Rolling and Expanding ##############

'calculate like rolling mean'; 'daily data is noise, so moving average better'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# -- Rolling (wondow average) -- movement
df = pd.read_csv('xxx/xxxx.csv',index_col='Date',parse_dates=True) 
df['Open'].plot(figsize=(16,6)) # plot time series
df.rolling(window=7).mean() # 7 days of moving average (Less noisy) \ > window = > soomth
'first 6 rows becomes NaN, first 7 = average of first 7 rows, then moving 1 by 1'

# compare two lines
df['Open'].plot() # plot open price
df.rolling(window=7).mean()['close'].plot() # plot moving averaged close price
# add to df, compare two lines
df['Close 30 days MA'] = df['Close'].rolling(window=30).mean()
df[['Close 30 days MA','Close']].plot(figszie=(16,6)) # plot two columns in pandas, it automatically add lengend


# -- Expanding (culmulative average) -- major trend
df['Close'].expanding().mean().plot(figsize=(16,6))


# -- Bollinger Bands (wider variance large, small variance small)
'price is high above the upper band, low below the lower band' - # not enecessary a sell or buy signal alone
# Close 20 moving averafe
df['Close: 20 Day Mean'] = df['Close'].rolling(20).mean()
# Upper band = Moving average(20) + 2 * Moving std(20)
df['Upper'] = df['Close: 20 Day Mean'] + 2*(df['Close'].rolling(20).std())
# Lower band = Moving average(20) - 2 * Moving std(20)
df['Lower'] = df['Close: 20 Day Mean'] - 2*(df['Close'].rolling(20).std())
# Close and plot
df[['Close','Close: 20 Day Mean','Upper','Lower']].plot(figsize=(16,6))



















'Stock Market Analysis -- Example'
--------------------------------------

########## Getting Data ##############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# -------- Getting data
import pandas_datareader
import datetime
import pandas_datareader.data as web

start = datetime.datetime(2012,1,1)
end = datetime.datetime(2017,1,1)


# ------- Getting other companies
tesla = web.DataReader('TSLA','google',start,end)
ford = web.DataReader('F','google',start,end)
gm = web.DataReader('GM','google',start,end)




######### Visualization - Spot Checks ################
# -- Plot three stocks together in lines to see differences
tesla['Open'].plot(label='Tesla',figsize=(12,8),title='Opening Prices')
gm['Open'].plot(label='GM')
ford['Open'].plot(label='Ford')
plt.legend()
# 'From this plot, Tesla may seems more valuable than others but to 
# get actual pic, we need to look at market cap of this company, not
# just the stock prices,so simple calculation -- 
# total money traded = volme X open prices 
# (100 units, $10 vs 100000 units, $1)


# market cap data added
tesla['Total Traded'] = tesla['Open']*tesla['Volume']
gm['Total Traded'] = gm['Open']*gm['Volume']
ford['Total Traded'] = ford['Open']*ford['Volume']
# plotting 'total traded' against time index
tesla['Total Traded'].plot(label='Tesla',figsize=(16,8))
gm['Total Traded'].plot(label='GM')
ford['Total Traded'].plot(label='Ford')
plt.legend();
# 'Tesla' is closer to others in amount of money been traded


# check spots of high money traded dates for each, correlated? 
# What happened that dates? Google it!
tesla['Total Traded'].argmax()

# -- Plot the volme of stock traded each day
tesla['Volume'].plot(label='Tesla',figsize=(12,8),title='Volume traded')
gm['Volume'].plot(label='GM')
ford['Volume'].plot(label='Ford')
plt.legend()
# 'Intereseting to see what happened in those high volume day of trading'

# 'Intereseting to see what happened in those high volume day of trading'
ford['Volume'].argmax() # Google news on that date for ford

# 'Check how up and down caused by those high volume trading event for ford'
ford['Open'].plot(figsize=(20,6))

# --- MA plottings MA50, MA200
gm['MA10'] = gm['Open'].rolling(10).mean()
gm['MA50'] = gm['Open'].rolling(50).mean()
gm[['Open','MA50','MA200']].plot(figsize=(16,6))


# --- Relationships between stocks
from pandas.tools.plotting import scatter_matrix
car_comp = pd.concat([tesla['Open'],gm['Open'],ford['Open']],axis=1)
car_comp.columns = ['Tesla Open','GM Open','Ford Open']
scatter_matrix(car_comp,figsize=(8,8),
               alpha=0.2,hist_kwds={'bins':50}) # alpha - darker = more density
                                                # hist_kwds - more bins - 50
# 'trends in diagnose, behavior of changes correlated pair stocks | GM more correlated to ford'


# ---- Plot candle sticks plots (short period is better)
from matplotlib.finance import candlestick_ohlc
from matplotlib.dates import DateFormatter, date2num, WeekdayLocator, DayLocator, MONDAY
ford_reset = ford.loc['2017-01'].reset_index() # get all Jan values and reset index
ford_reset.head()
ford_reset['date_ax'] = ford_reset['Date'].apply(lambda date: date2num(date))
ford_reset.head()
# Create list of tuples
list_of_cols = ['date_ax','Open','High','Low','Close']
ford_values = [tuple(vals) for vals in ford_reset[list_of_cols].values] 
mondays = WeekdayLocator(MONDAY) # Major ticks on the Monday
alldays = DayLocator() # minor ticks on that days
weekFormatter = DateFormatter('%b %d') # e.g. Jan 12
dayFormatter = DateFormatter('%d') # e.g. 12
# Plotting
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
# set index scales
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(weekFormatter)
candlestick_ohlc(ax,ford_values,width=0.1,colorup='g',colordown='r')




############ Basic Financial Analysis - Daily Return #############
# -- Daily Percentage Change
# * r(t) - return at time t, P(t) - price at time t
# r(t) = P(t)/P(t-1) - 1 --- If DPCs has a wide dist, more volatile, more reward/risk
#                        --- If DPCs has a concentrated dist, less volatile, less reward/risk
tesla['returns_manual'] = (tesla['Close'] / tesla['Close'].shift(1)) - 1
tesla['returns_auto'] = tesla['Close'].pct_change(1)
# Which stock is more volatile?
tesla['returns_auto'] = tesla['Close'].pct_change(1)
gm['returns_auto'] = gm['Close'].pct_change(1)
ford['returns_auto'] = ford['Close'].pct_change(1)
# Plot separately
tesla['returns_auto'].hist(bins=100)
gm['returns_auto'].hist(bins=100)
ford['returns_auto'].hist(bins=100)
# plot bar plots together
tesla['returns_auto'].hist(bins=100,label='Tesla',figsize=(10,8),alpha=0.4)
gm['returns_auto'].hist(bins=100,label='Tesla',figsize=(10,8),alpha=0.4)
ford['returns_auto'].hist(bins=100,label='Tesla',figsize=(10,8),alpha=0.4)
plt.legend()
# Plot density
# 'See clear' on violatile (Density plot)
tesla['returns_auto'].plot(kind='kde',label='Tesla',figsize=(10,8))
gm['returns_auto'].plot(kind='kde',label='Ford',figsize=(10,8))
ford['returns_auto'].plot(kind='kde',label='GM',figsize=(10,8))
plt.legend()
# Plot Boxplot
# See violatile on box plot
box_df = pd.concat([tesla['returns_auto'],ford['returns_auto'],gm['returns_auto']],axis=1)
box_df.columns = ['Tesla','Ford','GM']
box_df.plot(kind='box')

# (scatter plot on DPCs) See correlation between those 3, see how related the car companies are
scatter_matrix(box_df,figsize=(8,8),alpha=0.2,hist_kwds={'bins':100})
# 'Ford correlated to GM, Tesla not quit the same to others 






########### Basic Financial Analysis - Cumulative return ##############
# Cumulative return is the aggregated amount 
# of investment has gain or lost over time, 
# independent of the period of time involved.

# Why daily returns i going down? Start going down with time
'Date'   'Daily Return'   '%Daily Return'
'01'     '10/10 = 1'      '-'
'02'     '15/10 = 3/2'    '50%'
'03'     '20/15 = 4/3'    '33%'
'04'     '25/20 = 5/4'    '20%'

# Daily return is helpful but doesn't give investor immediate insight
# into the gains he or she had made till date, especially if the stock
# is very volatile.

# 'Cumulative Return' - above 1 profit, below one you are in loss
'Date'   'Cumulative Return'   '%Cumulative Return'
'01'     '10/10 = 1'           '100%'
'02'     '15/10 = 3/2'         '150%'
'03'     '20/10 = 2'           '200%'
'04'     '25/10 = 5/2'         '250%'

# i(i) = (1 + r(t)) * i(t-1)
tesla['Cumulative Return'] = (1 + tesla['returns_auto']).cumprod()
ford['Cumulative Return'] = (1 + ford['returns_auto']).cumprod()
gm['Cumulative Return'] = (1 + gm['returns_auto']).cumprod()
# '% loss/profit if invest at date No1"

tesla['Cumulative Return'].plot(label="Tesla",figsize=(16,8))
ford['Cumulative Return'].plot(label="ford",figsize=(16,8))
gm['Cumulative Return'].plot(label="gm",figsize=(16,8))
plt.legend()
# 'overtime, Tesla profit more, then gm, then ford

















'Time Series Modeling - for Trading'
--------------------------------------









