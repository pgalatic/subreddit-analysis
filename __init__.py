SUBREDDIT_NAMES = ['LegalAdvice', 'relationships']
ATTRIBUTES = ['selftext', 'title']

import numpy as np
from sklearn import metrics

def load_data(arg):
    """
    Loads the data. There are three different data sets--train, dev, and test.
    """
    NAMES = SUBREDDIT_NAMES
    # In order to ensure a consistent labeling, they are labeled according to 
    # their order in the NAMES list.
    classes = [x for x in range(len(NAMES))]
    if arg == 'train':
        filenames = ['data/' + name + '_train.json' for name in NAMES]
    elif arg == 'dev':
        filenames = ['data/' + name + '_dev.json' for name in NAMES]
    elif arg == 'test':
        filenames = ['data/' + name + '_test.json' for name in NAMES]
    else:
        print('Invalid argument: %s', arg)
        sys.exit(0)
    
    features = []
    labels = []
    
    # For each file, open it, and grab the text information for each post.
    # Label that information based on its class, and then add the sets to the
    # current feature/label lists.
    for idx in range(len(filenames)):
        with open(filenames[idx], 'r') as f:
            curr_features = []
        
            for line in f:
                # Not secure, but convenient.
                dict = eval(line)
                curr_features.append(dict['title'] + ' ' + dict['selftext'])
            
            curr_labels = [classes[idx]] * len(curr_features)
        
        features += curr_features
        labels += curr_labels
        
    return features, labels
    
def evaluate(model, features, labels):
    """
    Evaluates the model in terms of accuracy, precision, and recall.
    
    If a model returns a float "confidence" value, that value is rounded to
    either 0 or 1, whichever is closer.
    """
    predictions = model.predict(features)
    
    if 1 > predictions[0] > 0:
        predictions = np.where(predictions > 0.5, 1, 0)
        
    stats = metrics.precision_recall_fscore_support(
                y_true=labels, 
                y_pred=predictions,
                pos_label=1,
                average='binary'
        )
        
    precision = stats[0]
    recall = stats[1]
    
    print 'Accuracy:\t%1.4f' % np.mean(predictions == labels)
    print 'Precision:\t%1.4f' % precision
    print 'Recall:\t\t%1.4f' % recall
