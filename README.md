[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/V9O7PxYS)
# HG2051 (AY23/24) Project 2: Group assignment

##   Introduction

This project constitutes 30% of your final grade for HG2051. Please work on the
final program in groups of 2-3 and report together.

- The goal of this assignment is to demonstrate your programming and
problem-solving abilities through teamwork. If your team has an idea for
another project that you would like to do instead, talk to me for approval.

- This project involves text classification using machine learning tools. As
with the individual project, your team will be required to submit data, output,
and annotated code along with a short writeup that describes your goals,
process, and results. Your code will be assessed based on its functionality and
simplicity.

> Github supports code collaboration via pull requests and merging. To
facilitate this, you should designate one person from your team as the
administrator and let me know who that is. This person will be responsible for
approving merges/changes from each group member's individual branch to the master
branch.

## Project 2: Text classification

Text classification is a general term for automatically identifying properties
of a text based on its linguistic content. For this project you will develop a
program that classifies short SMS messages based on whether they are `spam` or
not (`ham`). The initial project code in this repository will help you to get
started, along the following lines:

1. The corpora you will primarily be using are located in the `datasets` subfolder, and a brief explanation of each are given here:

      `SMSSpamCollection.txt` - this is the full UCI dataset, which is compiled from various sources. It contains both `ham` and `spam` labeled messages. You can view and read more about the dataset [here](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

      `SMSSpamCollection_500.txt` - this is a set of the first 500 messages from the dataset above, and is used in the code below for testing.

      `sms_corpus_NUS-ham.txt` - this is the full SMS corpus developed by NUS Singapore for machine learning & NLP. All of these messages are actual messages collected in Singapore from study participants. I have reformatted the data to match the format of the other datasets. You can read more [here](https://github.com/kite1988/nus-sms-corpus).

      `email_corpus_lingspam.txt` - this is a set of real emails from the Linguist List (a listserve site for dissemination information about linguistics) along with spam emails. I have reformatted the data to match the other datasets, including `ham` and `spam` labels. This is the dataset you will be using for evaluation, and you should not use it for training your classifier. You can download the original dataset [here](http://nlp.cs.aueb.gr/software_and_datasets/lingspam_public.tar.gz). The zipped folder contains a `readme` file explaining the data.

The [SMS spam collection from UCI](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) will be the set of data that you will use for training and testing the classifier. Since each of the messages in this corpus has already been classified as `spam` or `ham`, this allows us to see how well a basic Naive Bayes NLTK classifier works.

2. Create a list of `spam` words and `ham` words based on the labels of the short texts. You can either use the same dataset of messages or another that you can find online. Once you have sorted and filtered the words, perhaps via frequency distributions, write a function that uses these words in conjunction with the NLTK classifier to classify the messages and improve on the base score (83% accuracy on emails). This process is known as feature selection/engineering - the goal of the project is to implement other "features" that you can identify in the text which might improve your classification. Some ideas for sentiment analysis of movie reviews can be found [here](https://realpython.com/python-nltk-sentiment-analysis/#selecting-useful-features). Keep in mind that successfully differentiating between real and fake messages/emails may require different features than for differentiating between the sentiment of movie reviews.

3. Once you have determined the features that you want to extract, write a function that automatically extracts these features from the text as a dictionary. Then use the function to create a list of tuples, where the first item of the tuple is the dictionary of features and the second item is the category/classification ("ham" or "spam") of the text.

4. Split the dataset for training and validation to see how/whether your features improve classification using, i.e. the Naive Bayes classification algorithm in NLTK. It is important that whatever classifier you use, the training data is kept separate from the testing/validation set, otherwise you are simply letting the model "memorize" inputs (overfitting) and it will not generalize well to other datasets.

5. Once you have a model that you think works pretty well on the SMS dataset, test its accuracy on the email dataset (which you have not used for training) to see how well it classifies email texts as ham/spam. At this point, you can train your classifier on the complete SMS dataset, since your email dataset is being used for testing (it is 'unseen'). Additionally, you can incorporate other datasets (like the complete NUS SMS dataset or other datasets you can find online) into your training data. If you do use other datasets, make sure to include them in the git repository (under the `datasets` folder) and include links to your sources in your code/writeup. Your team's ultimate goal is to improve the accuracy of your classifier on the unseen email dataset.

6. This code is simply a starting point for exploration. What other features can you think of and implement to improve accuracy? How does the accuracy improve with different splits? Try to implement other classifiers and see how they perform on this dataset. Are there other lexical resources/corpora or algorithms within NLTK or from other sources that can improve your results? What is the best accuracy you can get? How does the genre of text (sms/emails) affect your classifier's results?

7. With your group, write a roughly 5-page paper (no longer than 10 pages) describing your goals, data, process, and results, and discussing some of the concerns identified in point 6. This paper should include relevant linguistic information and citations for your sources. Submit the paper as a PDF along with your annotated/commented Python code in a Github repository. The PDF should also be submitted via TurnItIn.

## Output

Your final updated repository should include the following items:

1. A `README.md` file (replace this one) with a brief summary of the project and a list of the items in the repository along with a brief description. See [this](https://github.com/lingdoc/praatscripts) and [this](https://github.com/lingdoc/V1_AA_project_scripts/tree/main/Toolbox_scripts) for examples. The `.md` extension means the `README` file makes use of ['markdown'](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) code, which is commonly used for formatting text on GitHub and elsewhere.

2. A single python script or a set of scripts/folders containing code to build your classifier and assess its performance on the `email_corpus_lingspam` dataset. I should be able to run a single script and get a similar result to what you report in your paper and in the `output.txt` file. Bonus points if you use folders/modules in developing your process. This does not include the `datasets` folder, which should contain all of the data used to develop your classifier. **Your main script should contain all names of the group and matriculation numbers!**

3. An `output.txt` file that contains the accuracy results of 10 runs of your final classifier being trained on the SMS+ dataset and evaluated on the email corpus dataset, as well as the computed average of these 10 runs. Code to do this is found in the initial project script.

4. A PDF of your 5-10 page paper describing your goals, data, process, and results.

> We will have a leaderboard that tracks progress of the projects via your
`output.txt` file. I will update the leaderboard every week with your most
recent results, so feel free to commit/push your code/output changes regularly.
