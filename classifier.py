# author: Paul Galatic

import __init__
import os
import sys
import json
import pickle
import numpy as np
from sklearn import pipeline
from sklearn import linear_model
from sklearn.feature_extraction import text

def load_data(arg):
    NAMES = __init__.SUBREDDIT_NAMES
    classes = [x for x in range(len(NAMES))]
    if arg == 'train':
        filenames = ['data/' + name + '_train.json' for name in NAMES]
    elif arg == 'dev':
        filenames = ['data/' + name + '_dev.json' for name in NAMES]
    else:
        print('Invalid argument: %s', arg)
        sys.exit(0)
    
    features = []
    labels = []
    
    for fname in filenames:
        with open(fname, 'r') as f:
            curr_features = []
        
            for line in f:
                dict = eval(line)
                curr_features.append(dict['title'] + ' ' + dict['selftext'])
            
            curr_labels = [classes[idx]] * len(curr_features)
        
        features += curr_features
        labels += curr_labels
        
    return features, labels

def train():
    features, labels = load_data('train')
        
    vectorizer = text.CountVectorizer(decode_error='ignore', stop_words='english')
    transformer = text.TfidfTransformer()
    
    classifier = linear_model.SGDClassifier(
                    loss='hinge', 
                    penalty='l2',
                    alpha=1e-3,
                    tol=1e-3,
                    random_state=42)
    
    text_clf_svm = pipeline.Pipeline(steps=[('vect', vectorizer),
                                       ('tfidf', transformer),
                                       ('clf-svm', classifier)])
    
    text_clf_svm.fit(features, labels)
    
    predictions = text_clf_svm.predict(features)
    print('Base training accuracy: %d', np.mean(predictions == labels))

    return text_clf_svm
    
def dev(model):
    features, labels = load_data('dev')
    
    predictions = model.predict(features)
    
    print('Dev training accuracy: %d', np.mean(predictions == labels))

if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print('usage: svm.py [train|dev|test]')
        sys.exit(0)
    
    if sys.argv[1] == 'train':
        model = train()
        with open('models/svm-model.pkl', 'wb') as f:
            pickle.dump(model, f)
    elif sys.argv[1] == 'dev':
        if os.path.isfile('models/svm-model.pkl'):
            with open('models/svm-model.pkl', 'rb') as f:
                model = pickle.load()
        else:
            model = train()
        
        dev(model)
    else:
        print('Invalid argument: %s' % sys.argv[1])