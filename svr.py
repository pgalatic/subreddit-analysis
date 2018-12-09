# author: Paul Galatic
#
# Uses a support vector machine to classify Reddit posts.

import __init__
import os
import sys
import time
import pickle
import numpy as np
from sklearn import pipeline
from sklearn import svm
from sklearn.feature_extraction import text

MODEL_NAME = 'models/svr-model.pkl'

def train():
    """
    Builds the SVM based on training data.
    """
    features, labels = __init__.load_data('train')
        
    vectorizer = text.CountVectorizer(decode_error='ignore', stop_words='english')
    transformer = text.TfidfTransformer()
    
    classifier = svm.SVR(kernel='sigmoid', gamma='scale')
    
    # Serializes the processing steps that would be required of the above.
    text_clf = pipeline.Pipeline(steps=[('vect', vectorizer),
                                       ('tfidf', transformer),
                                       ('clf-svr', classifier)])
    
    start = time.time()
    text_clf.fit(features, labels)
    print 'Training time:\t%1.4f seconds' % (time.time() - start)
    
    __init__.evaluate(text_clf, features, labels)

    return text_clf
    
def dev(model):
    """Tests the SVM based on dev data."""
    features, labels = __init__.load_data('dev')
    
    __init__.evaluate(model, features, labels)

def test(model):
    """Tests the SVM based on test data."""
    features, labels = __init__.load_data('test')
    
    __init__.evaluate(model, features, labels)
    
def bin(model):
    """Uses binning to identify the posts the model was most uncertain with."""
    features, labels = __init__.load_data('train')
    
    features = [feature.replace('\n', '') for feature in features]
    
    predictions = model.predict(features)
    
    bins = [list() for x in range(10)]
    
    for idx in range(len(predictions)):
        curr = predictions[idx]
        if curr < 0.1:
            bins[0].append(str(labels[idx]) + ' : ' + features[idx] + '\n')
        elif curr < 0.2:
            bins[1].append(str(labels[idx]) + ' : ' + features[idx] + '\n')
        elif curr < 0.3:
            bins[2].append(str(labels[idx]) + ' : ' + features[idx] + '\n')
        elif curr < 0.4:
            bins[3].append(str(labels[idx]) + ' : ' + features[idx] + '\n')
        elif curr < 0.5:
            bins[4].append(str(labels[idx]) + ' : ' + features[idx] + '\n')
        elif curr < 0.6:
            bins[5].append(str(labels[idx]) + ' : ' + features[idx] + '\n')
        elif curr < 0.7:
            bins[6].append(str(labels[idx]) + ' : ' + features[idx] + '\n')
        elif curr < 0.8:
            bins[7].append(str(labels[idx]) + ' : ' + features[idx] + '\n')
        elif curr < 0.9:
            bins[8].append(str(labels[idx]) + ' : ' + features[idx] + '\n')
        else:
            bins[9].append(str(labels[idx]) + ' : ' + features[idx] + '\n')
    
    for idx in range(len(bins)):
        with open('bins/%1.1f.txt' % (idx / 10.0), 'w') as out:
            out.writelines(bins[idx])
    
if __name__ == '__main__':
    """
    Four options:
        - Build the model
        - Test the model (using dev set)
        - Test the model (using test set)
        - Make 'bins' of the data based on likelihood
    """
    if len(sys.argv) != 2:
        print('usage: svr.py [train|dev|test]')
        sys.exit(0)
    
    if sys.argv[1] == 'train':
        model = train()
        with open(MODEL_NAME, 'wb') as f:
            pickle.dump(model, f)
    elif sys.argv[1] == 'dev':
        if os.path.isfile(MODEL_NAME):
            with open(MODEL_NAME, 'rb') as f:
                model = pickle.load(f)
            dev(model)
        else:
            print 'Cannot perform dev test with no model. Run train first.'
            sys.exit(0)
    elif sys.argv[1] == 'test':
        if os.path.isfile(MODEL_NAME):
            with open(MODEL_NAME, 'rb') as f:
                model = pickle.load(f)
            test(model)
        else:
            print('Cannot perform test with no model. Run train first.')
            sys.exit(0)
    elif sys.argv[1] == 'bin':
        if os.path.isfile(MODEL_NAME):
            with open(MODEL_NAME, 'rb') as f:
                model = pickle.load(f)
            bin(model)
        else:
            print 'Cannot perform binning with no model. Run train first.'
            sys.exit(0)
    else:
        print('Invalid argument: %s' % sys.argv[1])
        sys.exit(0)
