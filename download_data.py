# author: Paul Galatic
#
# downloading data for intelligent systems project
#

import __init__
import os
import pdb
import time
import json
import pickle
from reddit import Reddit

TOTAL_POSTS = 10000
LIMIT = 100
SUBREDDIT_NAMES = __init__.SUBREDDIT_NAMES
ATTRIBUTES = __init__.ATTRIBUTES

class Data():
    def __init__(self):
        self.subreddit_names = SUBREDDIT_NAMES
        self.postset = set()

def grab_posts(dat, reddit):
    """
    Grabs the latest 100 "hot" posts from both subreddits.
    """
    
    subreddits = {name:reddit.subreddit(name) for name in dat.subreddit_names}
    post_groups = {}
    for key in subreddits:
        post_groups[key] = subreddits[key].hot(limit=LIMIT)
    
    items = []
    skipped = 0
    for key in post_groups:
        group = post_groups[key]
        with open(key + '.json', 'a') as f:
            for post in group:
                to_dict = vars(post)
                id = to_dict['id']
                # don't download duplicates
                if not id in dat.postset:
                    subset = {attr:to_dict[attr] for attr in ATTRIBUTES}
                    json.dump(subset, f)
                    f.write('\n')
                    dat.postset.add(id)
                else:
                    skipped += 1
                
    print('SKIPPED: ', skipped)

if __name__ == '__main__':
    """
    Runs in a loop. Every two hours, collects data, then sleeps.
    """
    while True:
        print('Collecting data...')
    
        # Does the data file exist? If not, create it.
        # The data file contains a list of posts we've already seen.
        if os.path.isfile('data.pkl'):
            with open('data.pkl', 'rb') as f:
                dat = pickle.load(f)
        else:
            dat = Data()
        
        # Create the Reddit instance.
        reddit = Reddit().reddit
        
        # Retrieve the data.
        grab_posts(dat, reddit)
        
        # Write the data file to disk.
        with open('data.pkl', 'wb') as f:
            pickle.dump(dat, f)
            
        time.sleep(7200)