## The following code gives some initial direction in building a spam detection
## tool using resources from the nltk library. The code use several publicly
## available datasets that have been processed into a common format for ease of
## use. You may find other online sources that can give you further ideas on how
## to improve on analyzing and detecting spam, and you are welcome to incorporate
## such code into your project.

import nltk
from nltk.corpus import stopwords, names
from statistics import mean
from random import shuffle
import pandas as pd
import re

####### Zhong Han's note:
####### Added a Lemmatizer so that we can lemmatize our SMS and email datasets
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

####### Zhong Han's note:
####### Type the following line into your VScode terminal 
####### $pip install -U scikit-learn

import sklearn
from sklearn.svm import SVC
from nltk.classify.scikitlearn import SklearnClassifier


## The corpora you will primarily be using are located in the `datasets`
## subfolder, and a brief explanation of each are given here:
## `SMSSpamCollection.txt` - this is the full UCI dataset, which is compiled
##        from various sources. It contains both `ham` and `spam` labeled
##        messages. You can view and read more about the dataset at:
##        https://archive.ics.uci.edu/dataset/228/sms+spam+collection
## `SMSSpamCollection_500.txt` - this is a set of the first 500 messages from
##        the dataset above, and is used in the code below for testing
## `sms_corpus_NUS-ham.txt` - this is the full SMS corpus developed by NUS Singapore
##        for machine learning & NLP. All of these messages are actual messages
##        collected in Singapore from study participants. I have reformatted the
##        data to match the format of the other datasets. You can read more at:
##        https://github.com/kite1988/nus-sms-corpus
## `email_corpus_lingspam.txt` - this is a set of real emails from the Linguist
##        List (a listserve site for dissemination information about linguistics)
##        along with spam emails. I have reformatted the data to match the
##        other datasets, including `ham` and `spam` labels. This is the
##        dataset you will be using for evaluation, and you should not use it
##        for training your classifier. You can download the original dataset at:
##        http://nlp.cs.aueb.gr/software_and_datasets/lingspam_public.tar.gz
##        The zipped folder contains a `readme` file explaining the data.

# Now let's take a look at the corpus we will use to test the classifier
sms = pd.read_table('datasets/SMSSpamCollection_500.txt', header=None, encoding='utf-8') # use this for testing/illustration because it is smaller
# sms = pd.read_table('datasets/SMSSpamCollection.txt', header=None, encoding='utf-8') # use this for the final version of the project
print(sms.head()) # print the first 5 lines of the dataframe
print(sms.info()) # print general info about the dataframe
print(sms[0].value_counts()) # print the labels located in the first column of the dataframe

# The corpus is a single text file with tab-delimited lines, where the first
# item in the line is the label ('spam' or 'ham') and the second item is the
# actual message.
# Above we read the messages into a pandas dataframe (`sms`), and we can further
# separate out the different kinds of messages by creating new dataframes below.
ham_sms = sms[sms[0]=='ham'] # get all lines where the first column has the label 'ham'
print("number of ham messages:", len(ham_sms)) # print our number (this should match output from line 48 above)
print(ham_sms.head()) # print the first 5 lines in the dataframe
spam_sms = sms[sms[0]=='spam'] # get all lines where the first column has the label 'spam'
print("number of spam messages:", len(spam_sms)) # print our number (this should match output from line 48 above)
print(spam_sms.head()) # print the first 5 lines in the dataframe
all_sms = ham_sms + spam_sms # combine the two dataframes in order, all 'ham' messages, then all 'spam' messages
print("number of total messages:", len(all_sms))

# We can now process the text in the messages.
# First let's remove the less 'contentful' words
unwanted = stopwords.words("english") # create a list of 'stopwords'
unwanted.extend([w.lower() for w in names.words()]) # extend this with a list of names
print("Number of 'unwanted' words:", len(unwanted)) # print the information to the terminal

# Let's now create a function to go through sentences tagged with Part-Of-Speech
# (POS) at the word level and skip them if they meet some criteria
def skip_unwanted(pos_tuple):
    word, tag = pos_tuple # for the two items in the pos-tagged tuple
    # check if the 'word' is alphabetic or in the `unwanted` list
    if not word.isalpha() or word in unwanted:
        # if it is, return the value `False`
        return False
    # also check if the 'word' is tagged as a noun (see https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk)
    if tag.startswith("NN"):
        # if so, return the value `False`
        return False
    # if neither of these conditions are met, keep the word
    return True

# We may be dealing with nested lists when using nltk's `pos_tag()` on
# paragraphs, so let's write a simple function to flatten these nested lists
def flatten(nested_list):
    flat = []
    for item in nested_list:
        if type(item) == list:
            for wd in item:
                flat.append(wd)
        else:
            flat.append(item)
    return flat

# Now let's use our function to create a list of words based on the words in the
# `spam` and `ham` categories of our corpus (using list comprehension).
# Here the function `filter()` uses our function to filter the output of the
# nltk `pos_tag()` function.

print("Getting the ham words...")
# first get the words found in the `ham` category
ham_words = [x for x in ham_sms[1].apply(nltk.word_tokenize)] # tokenize the `ham` words
ham_tags = [nltk.pos_tag(x) for x in ham_words] # tag the `ham` words
ham_flat = flatten(ham_tags) # flatten the nested lists of tuples (tagged words)
# now reduce the set to only words we want, using our function above
ham_words = [word.lower() for word, tag in filter(
    # first argument for `filter` is the function we created
    skip_unwanted,
    # second argument is the pos-tagged set of words from the `ham` category of text messages
    ham_flat
    )]

print("number of `ham` words:", len(ham_words))

print("Getting the `spam` words...")
# next get the words found in the `spam` category
spam_words = [x for x in spam_sms[1].apply(nltk.word_tokenize)] # tokenize the `spam` words
spam_tags = [nltk.pos_tag(x) for x in spam_words] # tag the `spam` words
spam_flat = flatten(spam_tags) # flatten the nested lists of tuples (tagged words)
# now reduce the set to only words we want, using our function above
spam_words = [word.lower() for word, tag in filter(
    # first argument for `filter` is the function we created
    skip_unwanted,
    # second argument is the pos-tagged set of words from the `neg` category
    # of movie reviews
    spam_flat
    )]

print("number of `spam` words:", len(spam_words))


####### Zhong Han's note:
####### I added a lemmatizer at this step!

# Initializing a Lemmatizer to lemmatize the words
lemmatizer = WordNetLemmatizer()

# Lemmatizing the words after tokenization for ham messages and spam messages
ham_words = [lemmatizer.lemmatize(word.lower()) for sentence in ham_tags for word, tag in sentence]
spam_words = [lemmatizer.lemmatize(word.lower()) for sentence in spam_tags for word, tag in sentence]

print("Getting the frequency distributions...")
# now let's create frequency distributions of each list of words
ham_fd = nltk.FreqDist(ham_words)
spam_fd = nltk.FreqDist(spam_words)

# get the set of words that are common to both
common_set = set(ham_fd).intersection(spam_fd)

# remove words from each frequency distribution that occur in common, as these
# don't meaningfully distinguish between the types of messages
for word in common_set:
    del ham_fd[word]
    del spam_fd[word]

####### Zhong Han's note:
####### I changed from "Top 100" to "Top 30". This resulted in an immediate boost of accuracy from 83% to 88%.
####### I suspect it's because the original "Top 100" most common words wasn't specific enough.

# create a dictionary of the top 30 words remaining in each frequency distribution
top_30_ham = {word for word, count in ham_fd.most_common(30)}
top_30_spam = {word for word, count in spam_fd.most_common(30)}

# We now have a set of features consisting of the top 30 words occurring uniquely
# in `ham` and `spam` messages.
print(top_30_ham)
print(top_30_spam)

# We can now define a function to count these 'features' (words) in each sentence
def extract_features(text):
    features = dict()
    word_count_ham = 0
    exclamation_count = 0  # initialize the count of exclamation points

    ####### Zhong Han's note:
    ####### I tried to expand the scope a bit further by using bigrams instead of single words.
    ####### This part of the code is basically combining bigrams with lemmatization.

    # Tokenize text into lemmatized bigrams
    lemmatized_bigrams = list(nltk.bigrams([lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(text)]))

    for bigram in lemmatized_bigrams:
        # Check if both words in the lemmatized bigram are in the top_30_ham list
        if bigram[0] in top_30_ham and bigram[1] in top_30_ham:
            word_count_ham += 1

    ####### Zhong Han's note:
    ####### Originally, the starter code only had the "word count" feature (i.e. the classifier uses "most common spam/ham words" as a feature)
    ####### I added a second feature, counting how many "exclamation points" were in the dataset. This boosted accuracy by 1%.
    
    # Count exclamation points using regular expression
    exclamation_count = len(re.findall(r'!', text))

    features["wordcount_ham"] = word_count_ham
    features["exclamation_count"] = exclamation_count  # include the count of exclamation points as a feature
    return features


print("Extracting features...")
# Now we can create a new list consisting of a dictionary of features for each
# text in the sms dataset.
# First we extract features for the `ham` messages:
features = [
    (extract_features(text), "ham")
    for text in ham_sms[1]
    # for text in tqdm(ham_sms[1]) # uncomment this line and comment out the line above to view progress
    ]

# Then we extend this to the `spam` messages so that all messages have the same
# number of features:
features.extend([
    (extract_features(text), "spam")
    for text in spam_sms[1]
    # for text in tqdm(spam_sms[1]) # uncomment this line and comment out the line above to view progress
    ])

print(len(features))

# Now that we have 'featurized' all our text, we can use these features to train
# our classifier.

####### Zhong Han's note:
####### So this is the part where we start using classifiers
####### Originally, the starter code used only NLTK's Naive Bayes (NB) Classifer
####### Anytime you see the variable "classifier", it refers to the original NLTK NB Classifier
####### However, I'm going to utilize an SVM Classifier from scikit-learn (sklearn), a library meant for machine learning (whereas NLTK is meant for NLP)
####### Find out more at this link: https://scikit-learn.org/stable/ 

# First we get the length of a quarter of the dataset
train_count = len(features) // 4

shuffle(features) # shuffle the dataset

####### Zhong Han's note:
####### NLTK's NB classifier:
print(F"Training NLTK's NB classifier on {train_count} samples")
# instantiate a classifier that trains on the first quarter of the dataset
classifier = nltk.NaiveBayesClassifier.train(features[:train_count])
# print out the 10 features that most clearly distinguish between positive/negative classes
classifier.show_most_informative_features(10)

####### Zhong Han's note:
####### sklearn's SVM classifier:
print(F"Training SVM classifier on {train_count} samples")
svm_classifier = SklearnClassifier(SVC())
svm_classifier.train(features[:train_count])

print(F"Assessing NLTK's NB classifier on {len(features)-train_count} samples")
# assess how accurate the trained classifier is on the last three quarters of the dataset
print("Accuracy of NLTK's NB classifier:", nltk.classify.accuracy(classifier, features[train_count:]))

print(F"Assessing SVM classifier on {len(features)-train_count} samples")
# assess how accurate the trained classifier is on the last three quarters of the dataset
accuracy_svm = nltk.classify.accuracy(svm_classifier, features[train_count:])
print("Accuracy of SVM classifier:", accuracy_svm)

# Now let's import an unseen dataset of different data that has also been
# labeled as spam/ham. This is email data from the Linguist List, which I have
# processed into the same format as our SMS data.

# First we load the data and print some info on it
emails = pd.read_table('datasets/email_corpus_lingspam.txt', header=None, encoding='utf-8')
print(emails.head())
print(emails.info())
print(emails[0].value_counts())

# Then we featurize it
efeatures = [
    (extract_features(text), label)
    for text, label in zip(emails[1], emails[0])
    # for text in tqdm(ham_sms[1]) # uncomment this line and comment out the line above to view progress
    ]
# Then we assess our classifier's accuract on the featurized texts
print(F"Assessing NLTK's NB classifier on {len(efeatures)} samples")
print("Accuracy of NLTK's NB classifier on email data:", nltk.classify.accuracy(classifier, efeatures[:]))

print(F"Assessing SVM classifier on {len(efeatures)} samples")
print("Accuracy of SVM classifier on email data:", nltk.classify.accuracy(svm_classifier, efeatures[:]))

# You may notice that every time you run this code, your final accuracy score
# will differ. This is because you are shuffling (randomizing) your set of
# features used to train the classifier. It is common practice in machine
# learning to score a classifier based on the average accuracy value over a
# set number of runs, typically around 10. There are also ways of ensuring
# consistency (replicability) between runs, but taking the average of 10 runs
# will suffice for our purposes here. It is also common practice to train on a
# larger portion of the dataset and validate on a smaller portion of the dataset,
# so feel free to experiment with different "splits". To train a more robust
# model, you should also use a technique known as "cross-validation", but for
# now we will be content with simply taking the average of 10 runs.

# The code below runs 10 iterations of the classifier and writes each score to
# a text file, along with a final average score (mean) across all 10 iterations.
with open('output.txt', 'w') as f:
    nb_iterations = []
    svm_iterations = []

    for x in list(range(10)):
        print("Writing iteration number: ", x)
        shuffle(features)  # shuffle the dataset

        # Train a new Naive Bayes classifier on all the sms data
        nb_classifier = nltk.NaiveBayesClassifier.train(features[:])

        # Check accuracy of Naive Bayes classifier on all the email data
        nb_accuracy = nltk.classify.accuracy(nb_classifier, efeatures[:])
        nb_iterations.append(nb_accuracy)

        # Train a new SVM classifier on all the sms data
        svm_classifier = SklearnClassifier(SVC())
        svm_classifier.train(features[:])

        # Check accuracy of SVM classifier on all the email data
        svm_accuracy = nltk.classify.accuracy(svm_classifier, efeatures[:])
        svm_iterations.append(svm_accuracy)

        f.write(f"Iteration {x + 1}: {nb_accuracy} (Naive Bayes Accuracy)\n")
        f.write(f"Iteration {x + 1}: {svm_accuracy} (SVM Accuracy)\n\n")

    f.write("\nMean accuracy of Naive Bayes Classifier: " + str(mean(nb_iterations)))
    f.write("\nMean accuracy of SVM Classifier: " + str(mean(svm_iterations)))

# Now that you understand the basics of how to build a classifier, continue to
# experiment with developing features that may help your classifier to
# distinguish better between spam and ham.