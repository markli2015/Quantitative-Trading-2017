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
plt.tight_layout() # adjust overlapping index

axes[0].plot(x,y)
axes[0].set_title('LARGER PLOT1')

axes[1].plot(y,x)
axes[1].set_title('LARGER PLOT2')






# >>> Figure Size and DPI
fig = plt.figure(figsize=(3,2),dpi=100) # dpi - dots per pixel (usually use default)





















'Data Source'
------------------------






























''
------------------------


