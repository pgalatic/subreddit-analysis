# author: Paul Galatic
#
# Uses a support vector machine to classify Reddit posts.

import __init__
import os
import sys
import json
import pickle
import numpy as np
from sklearn import pipeline
from sklearn import linear_model
from sklearn.feature_extraction import text

MODEL_NAME = 'models/svm-model.pkl'

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
    text_clf_svm = pipeline.Pipeline(steps=[('vect', vectorizer),
                                       ('tfidf', transformer),
                                       ('clf-svm', classifier)])
    
    text_clf_svm.fit(features, labels)
    
    predictions = text_clf_svm.predict(features)
    print 'Base training accuracy: %1.4f' % np.mean(predictions == labels)

    # TODO Support Vector Regression *
    # TODO Confidence function of text_clf_svm *
    # TODO Fake reddit posts with one word for each word in vocabulary
    # TODO Word embeddings
    # TODO Logistic regression with accuracy *
    return text_clf_svm
    
def dev(model):
    """Tests the SVM based on dev data."""
    features, labels = __init__.load_data('dev')
    
    predictions = model.predict(features)
    
    print 'Dev training accuracy: %1.4f' % np.mean(predictions == labels)

if __name__ == '__main__':
    """
    Three options:
        - Build the model
        - Test the model (using dev set)
        - Test the model (using test set)
    """
    if len(sys.argv) != 2:
        print('usage: svm.py [train|dev|test]')
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
    else:
        print('Invalid argument: %s' % sys.argv[1])
        sys.exit(0)
