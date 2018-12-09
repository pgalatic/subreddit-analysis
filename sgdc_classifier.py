# author: Paul Galatic
#
# Uses a linear SGDC classifier to classify Reddit posts.

import __init__
import os
import sys
import time
import pickle
import numpy as np
from sklearn import pipeline
from sklearn import linear_model
from sklearn.feature_extraction import text

MODEL_NAME = 'models/sgdc-model.pkl'

def train():
    """
    Builds the SVM based on training data.
    """
    features, labels = __init__.load_data('train')
        
    vectorizer = text.CountVectorizer(decode_error='ignore', stop_words='english')
    transformer = text.TfidfTransformer()
    
    classifier = linear_model.SGDClassifier(
                    loss='hinge', 
                    penalty='l2',
                    alpha=1e-3,
                    tol=1e-3,
                    random_state=42)
    
    # Serializes the processing steps that would be required of the above.
    text_clf = pipeline.Pipeline(steps=[('vect', vectorizer),
                                       ('tfidf', transformer),
                                       ('clf-sgdc', classifier)])
    
    start = time.time()
    text_clf.fit(features, labels)
    print 'Training time:\t%1.4f seconds' % (time.time() - start)
    
    __init__.evaluate(text_clf, features, labels)

    return text_clf
    
def dev(model):
    """Tests the classifier based on dev data."""
    features, labels = __init__.load_data('dev')
    
    __init__.evaluate(model, features, labels)

def test(model):
    """Tests the classifier based on test data."""
    features, labels = __init__.load_data('test')
    
    __init__.evaluate(model, features, labels)
    
if __name__ == '__main__':
    """
    Three options:
        - Build the model
        - Test the model (using dev set)
        - Test the model (using test set)
    """
    if len(sys.argv) != 2:
        print('usage: sgdc_classifier.py [train|dev|test]')
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
            print('Cannot perform dev test with no model. Run train first.')
            sys.exit(0)
    elif sys.argv[1] == 'test':
        if os.path.isfile(MODEL_NAME):
            with open(MODEL_NAME, 'rb') as f:
                model = pickle.load(f)
            test(model)
        else:
            print('Cannot perform test with no model. Run train first.')
            sys.exit(0)
    else:
        print('Invalid argument: %s' % sys.argv[1])
        sys.exit(0)
