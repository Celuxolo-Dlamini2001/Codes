#!/usr/bin/env python
# coding: utf-8

# # Module 3 Unit 2
# ## Pruning a tree-based model 
# ### Classification problem
# Requiring a minimum number of samples in each leaf node dramatically prunes and simplifies a tree.  
# **Hint:** You can also explore the `min_samples_split` and other parameters for different pruning approaches. Refer to the enrichment activity at the end of this unit.

# In[1]:


# Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np 


# In[ ]:


# Import data
df = pd.read_csv('spam_uci.csv', delimiter = ",")


# In[ ]:


# Explore the size of the data set
df.shape


# In[ ]:


# Explore the types of data and feature names
df.head()


# ### Metadata
# 
# | Variable | Description
# | :--------|:-----------------------------------------------------------------|
# | crl.tot  | Total number of capital letters in the e-mail                    |
# | dollar   | Percentage of characters in the e-mail that match the “$” symbol |
# | bang	   | Percentage of characters in the e-mail that match the “!” symbol    |   
# | money	   | Percentage of words in the e-mail that match the word “money”  |
# | n000	   | Percentage of strings in the e-mail that match the string “000”  |
# | make	   | Percentage of words in the e-mail that match the word “make”   |
# | yesno	   | “n” for not spam and “y” for spam          |
# 
# The `yesno` variable will be the response variable.
# 

# In[ ]:


# Explore further
print("Number of emails classified as spam:",len(df[df.yesno == 'y']))
df.sample(10, random_state=0)


# In[ ]:


# Split data into features (X) and response (y)
# Given that the response is categorial, this is a classification problem
# Note that the column names need to have quotation marks
X = df.iloc[:, 1:7]
y = df.loc[:, ["yesno"]]


# In[ ]:


X.head()


# In[ ]:


y.head()


# Before training the model, it will be split into a training and test set, so the accuracy of the model can be determined before pruning.

# In[ ]:


# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(X_train.shape,X_test.shape)

# Fit data to tree-based classification model
classifier = DecisionTreeClassifier(random_state=0) 
classifier.fit(X_train, y_train)

# View accuracy prior to pruning
y_pred = classifier.predict(X_test)
test_score = accuracy_score(y_test, y_pred)

# Print the test score
print("Accuracy score of the tree = {:2.2%}".format(test_score)) 


# A model fitted without constraints has an accuracy of 86.10%. The model is likely overfitted on the data, with terminal nodes having as little as one sample.

# In[ ]:


# Plot of full tree
plt.figure()
plot_tree(classifier,feature_names=X.columns)
plt.show()


# In[ ]:


print("Tree depth =",classifier.get_depth(),'\n'
      "Number of leaves =",classifier.get_n_leaves())


# To prevent the model from overfitting, the number of samples in each terminal node or leaf can be restricted.
# 
# 
# 
# 

# In[ ]:


# Plot a more simplified tree, requiring a minimum of 40 samples in each 
# terminal node or leaf
classifier_small = DecisionTreeClassifier(random_state=0, 
                                          min_samples_leaf=40) 
# The default for min_samples_leaf is 1
classifier_small.fit(X_train, y_train)
plt.figure()
plot_tree(classifier_small,feature_names=X.columns)
plt.show()
print("Number of leaves =",classifier_small.get_n_leaves())


# So how do you determine the optimal tree size? That is, how do you decide on the minimum number of samples you want in each terminal node?

# In[ ]:


# Finding the optimal number of samples per leaf
samples = [sample for sample in range(1,50)]     

classifiers = []
for sample in samples:
    classifier2 = DecisionTreeClassifier(random_state=0, 
                                         min_samples_leaf=sample)
    classifier2.fit(X_train, y_train)
    classifiers.append(classifier2)


# In[ ]:


# Visualise the performance of each subtree on the training and test sets
train_scores = [clf.score(X_train, y_train) for clf in classifiers]
test_scores = [clf.score(X_test, y_test) for clf in classifiers]

fig, ax = plt.subplots()
ax.set_xlabel("Minimum leaf samples")
ax.set_ylabel("Accuracy")
ax.set_title("Comparing the training and test set accuracy")
ax.plot(samples, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(samples, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()


# In[ ]:


# Visualise how drastically the tree gets pruned by increasing the minimum 
# required samples per terminal node
nr_leaves = [clf.get_n_leaves() for clf in classifiers]

plt.figure()
plt.xlabel("Minimum required samples per terminal node")
plt.ylabel("Number of leaves")
plt.title("Number of tree leaves for min. samples per terminal node")
plt.plot(samples,nr_leaves)
plt.show()


# In[ ]:


# In order to find the optimal minimum leaf samples, cross validation is applied
validation_scores = []
for sample in samples:
    classifier3 = DecisionTreeClassifier(random_state=1, min_samples_leaf=sample)
    score = cross_val_score(estimator=classifier3, X=X_train, y=y_train, cv=5)   
    validation_scores.append(score.mean())


# In[ ]:


# Visualise the validation score in relation to minimum leaf samples.
plt.figure()
plt.xlabel("Minimum leaf samples")
plt.ylabel("Validation score")
plt.title("Validation scores at different minimum leaf sample counts")
plt.plot(samples, validation_scores, marker='o', label="train",
        drawstyle="steps-post")
plt.legend()
plt.show()


# In[ ]:


# Obtain the minimum leaf samples with the highest validation score.
samples_optimum = samples[validation_scores.index(max(validation_scores))]
print(samples_optimum)


# A minimum sample of 23 in terminal nodes has the best performance on the test set.

# In[ ]:


# Use the optimum  minimun leaf samples to fit a parsimonious tree
classifier4 = DecisionTreeClassifier(random_state=0, min_samples_leaf=samples_optimum)
classifier4.fit(X_train, y_train)


# In[ ]:


# Visualise the smaller pruned tree
plt.figure()
plot_tree(classifier4, feature_names=X_train.columns)
plt.show()

# Show the first few levels of the tree
plt.figure(figsize=[6,3], dpi=300)
plot_tree(classifier4, max_depth=2, 
          feature_names=X_train.columns, impurity=False)
plt.show()


# As you can see in the previous image, the tree is far less complex than the fully (most likely overfitted) tree, which had no minimum constraint on the number of samples in each leaf node. 

# By zooming in on the first couple of layers, you can see that the first split is made on the `!` or bang symbol. For interpretation of the decision tree, recall branches to the left indicate that the previous node condition was true and branches to the right mean the previous node condition was false.

# In[ ]:


# Final test to see how the model performs:
y_pred = classifier4.predict(X_test)
test_score2 = accuracy_score(y_test, y_pred)
print("Accuracy score of the optimal tree = {:2.2%}".format(test_score2)) 
print("Tree depth =",classifier4.get_depth(),'\n'
      "Number of leaves =",classifier4.get_n_leaves())


# The test accuracy is approximately 86.45%. This is only a minor increase in performance; however, in certain scenarios, small improvements are valuable. In most cases, performance will increase more dramatically after pruning.

# In[ ]:


# The final model to be used for predictions in the future:
best_model = DecisionTreeClassifier(random_state=0, min_samples_leaf=23)
best_model.fit(X, y)
print("Tree depth =",best_model.get_depth(),'\n'
      "Number of leaves =",best_model.get_n_leaves())


# ## Enrichment
# 
# Apart from the minimum required samples per terminal node, illustrated in the previous example, there are many other ways to prune a tree. For more information on pruning, explore the scikit-learn DecisionTreeClassifier documentation.
# Good pruning parameters to explore are:
# 1. *max_depth*: The maximum depth a tree should grow. (Note: this could be called a *'hedge'* pruning approach, as opposed to the *'fruit tree'* pruning approach we have used before.)
# 1. *max_leaf_nodes*: The maximum number of leaves a tree should have.
# 1. *min_impurity_decrease*: A minimum amount by which the impurity should decrease before adding a new split.
# 1. *min_samples_split*: A minimum number of samples a leaf should have before splitting it further.
# 

# Continue to the small group discussion to discuss the importance of pruning.
