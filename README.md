AUTHOR: Paul Galatic pgalatic@gmail.com

NOTE: Accuracies reported during your testing may not be exactly the same as those in the pdf report because the data will have been updated slightly. However, they should be similar.

# Subreddit Analysis

### /r/LegalAdvice word cloud

![LegalAdvice word cloud](https://github.com/pgalatic/subreddit-analysis/blob/master/LegalAdvice.png)

### /r/relationships word cloud

![relationships word cloud](https://github.com/pgalatic/subreddit-analysis/blob/master/Relationships.png)

## Setup

In order to continue downloading data, run the following program:

```
python download_data.py
```

It will query [/r/LegalAdvice](reddit.com/r/LegalAdvice) and [/r/relationships](reddit.com/r/relationships) every two hours and download the title+selftext of any post it has not already seen.

Before you run a classifier, you must split the data with this command:

```
python split_data.py
```

## Building Classifiers

In order to run any of the classifiers, use this style of command:

```
python [classifier] [train|dev|test]
```

* Train creates the model, if it doesn't already exist.
* Dev evaluates the model with dev data.
* Test evaluates the model with test data.

For example, if I wanted to train the logistic regression classifier, I would use this command:

```
python logistic_regression.py train
```

## Word Clouds

In order to create word clouds, use the command:

```
python word_cloud.py
```
