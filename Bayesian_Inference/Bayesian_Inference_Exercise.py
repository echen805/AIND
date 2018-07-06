'''
Bayesian_Inference Exercise to build Spam Classifier
'''
import pandas as pd
import string
import re


# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
name = ['label','sms_message']
df = pd.read_table('.\smsspamcollection\SMSSpamCollection', sep='\t', header= None, names = name)

# Output printing out first 5 rows
# print(df.head())
df.head()

df['label'] = df.label.map({'ham': 0, 'spam': 1})
# print(df.head())

# Implement Bag of Words from scratch
documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_documents = []
for i in documents:
    lower_case_documents.append(i.split(" "))
# print(lower_case_documents)

# Removes punctuation and tokenizes the input.
# Tokenizing a sentence in a document set means splitting up a sentence into individual words using a delimiter. The delimiter specifies what character we will use to
# identify the beginning and the end of a word(for example we could use a single space as the delimiter for identifying words in our document set.)
sans_punctuation_documents = []
regex = re.compile('[%s]' % re.escape(string.punctuation))
for individual_list in lower_case_documents:
    for words in individual_list:
        sans_punctuation_documents.append(regex.sub('', words))

# print(sans_punctuation_documents)

# Counts the frequency of each word
frequency_list = []
import pprint
from collections import Counter
frequency_list = Counter()

for i in sans_punctuation_documents:
    frequency_list[i] += 1

# pprint.pprint(frequency_list)

'''
Here we will look to create a frequency matrix on a smaller document set to make sure we understand how the 
document-term matrix generation happens. We have created a sample document set 'documents'.
'''
documents = ['Hello, how are you!',
                'Win money, win from home.',
                'Call me now.',
                'Hello, Call hello you tomorrow?']
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
count_vector.fit(documents)
count_vector.get_feature_names()

# Instructions: Create a matrix with the rows being each of the 4 documents, and the columns being each word. The corresponding (row, column) value is the frequency of
# occurrence of that word(in the column) in a particular document(in the row). You can do this using the transform() method and passing in the document data set as the
# argument. The transform() method returns a matrix of numpy integers, you can convert this to an array using toarray(). Call the array 'doc_array'

doc_array = count_vector.transform(documents).toarray()
# doc_array

# Instructions: Convert the array we obtained, loaded into 'doc_array', into a dataframe and set the column names to the word names(which you computed earlier using
# get_feature_names(). Call the dataframe 'frequency_matrix'.

frequency_matrix = pd.DataFrame(doc_array,columns=count_vector.get_feature_names())
frequency_matrix

'''
Solution

NOTE: sklearn.cross_validation will be deprecated soon to sklearn.model_selection 

Instructions: Split the dataset into a training and testing set by using the train_test_split method in sklearn. Split the data using the following variables:

X_train is our training data for the 'sms_message' column.
y_train is our training data for the 'label' column
X_test is our testing data for the 'sms_message' column.
y_test is our testing data for the 'label' column Print out the number of rows we have in each our training and testing data.

'''
# split into training and testing sets
# USE from sklearn.model_selection import train_test_split to avoid seeing deprecation warning.
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))