
# coding: utf-8

# In[2]:

import numpy as numpy
import pandas as pds
import seaborn as sen
import matplotlib.pyplot as mplt
get_ipython().magic('matplotlib inline')


# #### Reading dataset

# In[3]:

data = pds.read_csv('../data/dataset.csv')
data.head(10)


# ### Correlations analyze

# In[17]:

sen.set(style="white")

# Calculate correlation matrix
corr = data.corr()

# Generate a mask for the upper triangle
mask = numpy.zeros_like(corr, dtype=numpy.bool)
mask[numpy.triu_indices_from(mask)] = True

# Matplotlib figure setting
f, ax = mplt.subplots(figsize=(5, 4))

# Diverging colormap generate
cmap = sen.diverging_palette(10, 220, as_cmap=True)

# Output heatmap with the mask and correct aspect ratio
sen.heatmap(corr, mask=mask, cmap=cmap, vmax=.5,
            square=True, xticklabels=True, yticklabels=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


# ### Features analyze 

# In[19]:

sen.set(style="white")
f, ax = mplt.subplots(figsize=(7, 5))
sen.barplot(x=data.satisfaction_level,y=data.left,orient="h", ax=ax)


# In[6]:

sen.set(style="darkgrid")
g = sen.FacetGrid(data, row="department", col="left", margin_titles=True)
bins = numpy.linspace(0, 1, 13)
g.map(mplt.hist, "satisfaction_level", color="steelblue", bins=bins, lw=0)


# ### Employee leavers analysis

# In[7]:

sen.set(style="white", palette="muted", color_codes=True)

# Matplotlib figure setting
f, axes = mplt.subplots(3, 3, figsize=(9,7))
sen.despine(left=True)

#Left employees
leavers = data.loc[data['left'] == 1]

# Plot a simple histogram with binsize determined automatically
sen.distplot(leavers['satisfaction_level'], kde=False, color="b", ax=axes[0,0])
sen.distplot(leavers['salary_level'], bins=3, kde=False, color="b", ax=axes[0, 1])
sen.distplot(leavers['average_monthly_hours'], kde=False, color="b", ax=axes[0, 2])
sen.distplot(leavers['number_projects'], kde=False, color="b", ax=axes[1,0])
sen.distplot(leavers['last_evaluation'], kde=False, color="b", ax=axes[1, 1])
sen.distplot(leavers['time_spent_company'], kde=False, bins=5, color="b", ax=axes[1, 2])
sen.distplot(leavers['promotion_last_5_years'],bins=10, kde=False, color="b", ax=axes[2,0])
sen.distplot(leavers['work_accident'], bins=10,kde=False, color="b", ax=axes[2, 1])


mplt.tight_layout()


# ### Filter key employees

# In[8]:

#all key employees
key_employees = data.loc[data['last_evaluation'] > 0.7].loc[data['time_spent_company'] >= 3]
key_employees.describe()


# In[9]:

#lost key employees
lost_key_employees = key_employees.loc[data['left']==1]
lost_key_employees.describe()


# In[10]:

print ("Key employees count: ", len(key_employees))
print ("Lost key employees count: ", len(lost_key_employees))
print ("Lost key employees as percentage: ", round((float(len(lost_key_employees))/float(len(key_employees))*100),2),"%")


# In[11]:

#save key employees to csv
key_employees.to_csv('../data/key_employees.csv')


# ### Analyze performing employees leave

# In[12]:

#filter emplyees with good last evaluation
leaving_performers = leavers.loc[leavers['last_evaluation'] > 0.7]


# In[13]:

sen.set(style="white")

# Correlation matrix
corr = leaving_performers.corr()

# Generate a mask for the upper triangle
mask = numpy.zeros_like(corr, dtype=numpy.bool)
mask[numpy.triu_indices_from(mask)] = True

# Matplotlib figure setting
f, ax = mplt.subplots(figsize=(5, 4))

# Colormap
cmap = sen.diverging_palette(10, 220, as_cmap=True)

# Draw the heatmap
sen.heatmap(corr, mask=mask, cmap=cmap, vmax=.5,
            square=True, xticklabels=True, yticklabels=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


# ### Analyze satisfied employees leave

# In[14]:

#filter employees with a good satisfaction level
satisfied_employees = data.loc[data['satisfaction_level'] > 0.7]


# In[15]:

sen.set(style="white")

# correlation matrix
corr = satisfied_employees.corr()

# Generate a mask for the upper triangle
mask = numpy.zeros_like(corr, dtype=numpy.bool)
mask[numpy.triu_indices_from(mask)] = True

# matplotlib figure
f, ax = mplt.subplots(figsize=(5, 4))

# colormap
cmap = sen.diverging_palette(10, 220, as_cmap=True)

# Draw the heatmap
sen.heatmap(corr, mask=mask, cmap=cmap, vmax=.5,
            square=True, xticklabels=True, yticklabels=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

