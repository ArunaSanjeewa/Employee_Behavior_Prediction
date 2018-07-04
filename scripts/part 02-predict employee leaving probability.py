
# coding: utf-8

# In[44]:

import pandas as pds
import numpy as numpy
import warnings
warnings.filterwarnings('ignore')


# In[45]:

# load dataset
leave_df = pds.read_csv('../data/dataset.csv')
column_names = leave_df.columns.tolist()

#show attribute list
print ("Column names(attribute):\n")
print (column_names)

#print  first 10 rows
print ("\nSample 10 raws data:")
leave_df.head(10)


# In[46]:

# Get 'left' attribute data
y = leave_df['left']

# Remove unwanted columns
to_drop = ['name', 'salary', 'left']
leave_feat_space = leave_df.drop(to_drop,axis=1)

# Pull out features for future use
features = leave_feat_space.columns

# convert label features to integers
from sklearn import preprocessing
le_sales = preprocessing.LabelEncoder()
le_sales.fit(leave_feat_space["department"])
leave_feat_space["department"] = le_sales.transform(leave_feat_space.loc[:,('department')])

# transforme the whole feature space into a matrix
X = leave_feat_space.as_matrix().astype(numpy.float)

# standardize all features
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

print ("Feature space holds %d observations and %d features" % X.shape)
print ("Unique target labels:", numpy.unique(y))


# In[47]:

# prediction fun
def run_cv(X,y,clf_class, method, **kwargs):
    
    from sklearn.model_selection import cross_val_predict
    
    # Initialize a classifier with key word arguments
    clf = clf_class(**kwargs)
    
    predicted = cross_val_predict(clf, X, y, cv=3, method=method)
    
    return predicted


# In[48]:

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import metrics

def accuracy(y, predicted):
    # NumPy interprets True and False as 1. and 0.
    return metrics.accuracy_score(y, predicted)

print ("Support vector machines:")
print ("%.3f" % accuracy(y, run_cv(X,y,SVC, method='predict')))
print ("Random forest:")
print ("%.3f" % accuracy(y, run_cv(X,y,RF, method='predict')))
print ("K-nearest-neighbors:")
print ("%.3f" % accuracy(y, run_cv(X,y,KNN, method='predict')))


# In[49]:

from sklearn.metrics import confusion_matrix

y = numpy.array(y)
class_names = numpy.unique(y)

# calculate confusion matrices
confusion_matrices = [
    ( "Support Vector Machines", confusion_matrix(y,run_cv(X,y,SVC, method='predict')) ),
    ( "Random Forest", confusion_matrix(y,run_cv(X,y,RF, method='predict')) ),
    ( "K-Nearest-Neighbors", confusion_matrix(y,run_cv(X,y,KNN, method='predict')) ),
]

# show confusion matrix values
print (confusion_matrices)


# In[50]:

import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().magic('matplotlib inline')

# draw confusion matrices
for cf in confusion_matrices:

    ax = plt.axes()
    ax.set_title(cf[0])
    
    df_cm = pds.DataFrame(cf[1], index = ["Real 0", "Real 1"], columns = ["Pred 0", "Pred 1"])
    plt.figure(figsize = (6,5))
    sn.heatmap(df_cm, annot=True, ax = ax, square=True, fmt="d",linewidths=.5)


# In[51]:

# Use 10 estimators so predictions are all multiples of 0.1
pred_prob = run_cv(X, y, RF, n_estimators=10,  method='predict_proba',)

pred_leave = pred_prob[:,1]
is_leave = y == 1

# Number of times a predicted probability is assigned to an observation
counts = pds.value_counts(pred_leave)

# calculate true probability
true_prob = {}
for prob in counts.index:
    true_prob[prob] = numpy.mean(is_leave[pred_leave == prob])
    true_prob = pds.Series(true_prob)

# pandas-fu
counts = pds.concat([counts,true_prob], axis=1).reset_index()
counts.columns = ['pred_prob', 'count', 'true_prob']
counts


# In[52]:

#create a colums
pred_prob_df = pds.DataFrame(pred_prob)
pred_prob_df.columns = ['prob_not_leaving', 'prob_leaving']

#merge data
all_employees_pred_prob_df = pds.concat([leave_df, pred_prob_df], axis=1)

#filter employees work on the company and having a good evaluation
good_employees_still_working_df = all_employees_pred_prob_df[(all_employees_pred_prob_df["left"] == 0) & 
                                                            (all_employees_pred_prob_df["last_evaluation"] >= 0.7)]

good_employees_still_working_df.sort_values(by='prob_leaving', ascending=False, inplace=True)

#write to csv
good_employees_still_working_df.to_csv("../output/good_employees_leaving_prob.csv")


# In[ ]:



