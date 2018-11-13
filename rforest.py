# author: Paul Galatic
#
# Uses a random forest to classify Reddit posts.

import __init__
import os
import sys
from sklearn import ensemble
from sklearn import pipeline
from sklearn.feature_extraction import text

MODEL_NAME = 'models/rf-model.pkl'

def train():
    """Builds the random forest based on training data."""
    features, labels = __init__.load_data('train')
    
    vectorizer = text.CountVectorizer(decode_error='ignore', stop_words='english')
    transformer = text.TfidfTransformer()
    classifier = ensemble.RandomForestClassifier()

    text_clf_rf = pipeline.Pipeline(steps=[('vect', vectorizer),
                                       ('tfidf', transformer),
                                       ('clf-rf', classifier)])
    
    text_clf_rf.fit(features, labels)

    predictions = classifier.predict(features)

    print 'Base training accuracy: %1.4f' % np.mean(predictions == labels)

    return classifier

def dev(model):
    """Tests the random forest based on dev data."""
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