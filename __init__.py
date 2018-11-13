SUBREDDIT_NAMES = ['LegalAdvice', 'relationships']
ATTRIBUTES = ['selftext', 'title']

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