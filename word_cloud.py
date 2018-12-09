
import __init__
import os
import pdb
import wordcloud
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.feature_extraction import stop_words


if __name__ == '__main__':

    # Load ALL the data.
    train_features, train_labels =  __init__.load_data('train')
    dev_features, dev_labels =      __init__.load_data('dev')
    test_features, test_labels =    __init__.load_data('test')
    
    features = train_features + dev_features + test_features
    labels = train_labels + dev_labels + test_labels
    
    # Need to break up the data by label and sort it into bins.
    curr_label = labels[0]
    idx = 0
    feature_lists = [list() for _ in np.unique(labels)]
    while idx < len(labels):
        feature_lists[labels[idx]].append(features[idx])
        idx += 1
    
    # Make a word cloud for each bin.
    cloud_list = []
    for feature_list in feature_lists:
        all_text = ' '.join(feature_list)
        all_text = all_text.replace('u2019', '').replace('thing', '').replace('x200B', '')
        cloud = wordcloud.WordCloud(
                        background_color='white',
                        height=1000,
                        width=1600
                    ).generate(all_text)
        cloud_list.append(cloud)
    
    # Display each cloud.
    for cloud in cloud_list:
        plt.imshow(cloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
    