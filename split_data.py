# author: Paul Galatic

import __init__
import re
import json
import pickle

SUBREDDIT_NAMES = __init__.SUBREDDIT_NAMES
ATTRIBUTES = __init__.ATTRIBUTES

if __name__ == '__main__':
    """
    Splits data into train, test, and development sets.
    """
    
    for subreddit in SUBREDDIT_NAMES:
        with open(subreddit + '.json', 'r') as f:
            data = [eval(line) for line in f]
        
        train_set_idx = len(data) / 2
        test_set_idx = len(data) / 2 + len(data) / 4
        
        train_set = data[:train_set_idx]
        test_set = data[train_set_idx:test_set_idx]
        dev_set = data[test_set_idx:]
        
        with open('data/' + subreddit + '_train.json', 'w+') as f:
            for item in train_set:
                json.dump(item, f)
                f.write('\n')
        with open('data/' + subreddit + '_test.json', 'w+') as f:
            for item in test_set:
                json.dump(item, f)
                f.write('\n')
        with open('data/' + subreddit + '_dev.json', 'w+') as f:
            for item in dev_set:
                json.dump(item, f)
                f.write('\n')
        