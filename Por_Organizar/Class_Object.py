#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 09:49:02 2018

@author: MarioAntao
"""
import pickle
import random
import numpy as np
import pandas as pd
class Review(object):
    """Class represents tweet from Twitter.

    Attributes:
        reviewer_id: A unique identifier.
        product_id : A unique identifier.
        ReviewTime: A date-time, describing when the review was created.
        reviewText: A string, containing the text of the review.
        overall: A float, stating the rating of the product.
        summary: A string, containing the summary of the review.
        helpful: An integer, stating the number of favorites.
        category: A list, containing the helpfulness rating of the review.
    """

    def __init__(self, reviewer_id, product_id, ReviewTime, reviewText, overall, summary, helpful, category):
        """Initialzes Review class with defined content."""
        self.reviewer_id = reviewer_id
        self.product_id = product_id
        self.ReviewTime = ReviewTime
        self.reviewText = reviewText
        self.overall = overall
        self.summary = summary
        self.helpful = helpful
        self.category = category
        self.cleaning_log = dict()
    def __str__(self):
        """Creates user-friendly string representation of review."""

        return "reviewer_id: {}\nProduct_id: {}\nReviewTime: {}\nreviewText: {}\noverall: {}\nsummary: {}\nhelpful: {}\ncategory: {}".\
            format(self.reviewer_id, self.product_id, self.ReviewTime, self.reviewText, self.overall,
                   self.summary, self.helpful, self.category)

    def __eq__(self, other):
        """
        Determines if review is equal to other review.

        Compares the reviewer_id value. Returns True if reviewer_ids are equal.
        """
        return self.reviewer_id == other.reviewer_id

    def __hash__(self):
        """
        Returns a unique hash value.
        """
        return hash(self.reviewer_id)



    def count_number_of_words(self):
        """
        Counts the words in the review.

        Retrieves the text from the week_3.tweet, splits it by spaces and counts
        the length of the list.

        Returns:
            An interger value, corresponding to the number of words in the week_3.tweet.
        """
        word_list = self.reviewText.split()
        return len(word_list)



def create_review_from_dict(review_dict):
    """
    Creates a review object from dictionary.
    
    Extracts reviewer_id, helpful, reviewText, overall,
    summary,unixReviewTime, product_id and category from dictionary.
    
    Args:
        review_dict: A dictionary, containing review information.
        
    Returns:
        A review object.
    """
    
    # Extract parameters from dictionary
    reviewer_id = review_dict.get('reviewer_id')
    helpful = review_dict.get('helpful')
    reviewText = review_dict.get('reviewText')
    overall = review_dict.get('overall')
    summary = review_dict.get('summary')
    ReviewTime = review_dict.get('unixReviewTime')
    category = review_dict.get('category')
    product_id = review_dict.get('product_id')
    
    # Create review object
    review = Review(reviewer_id, product_id, ReviewTime, reviewText, overall, summary, helpful, category)
    
    return review

def make_review_set(filename):
    """
    Creates set of tweets from week_3.data/01_raw/trump_tweets.json.
    Returns:
        Set of week_3.tweet objects.
    """
    
    # Create set and review.Review objects.
    review_set = set()
      
    for revieww in sample_review:
        review = create_review_from_dict(revieww)
        review_set.add(review)

    return review_set


import pickle
import pandas as pd

merged_dic=pd.read_pickle('merged_dic.pickle')

sample_review=random.sample(merged_dic,1000)





review_set=make_review_set(merged_dic)

review_train_set=make_review_set(review_train)
review_test_set=make_review_set(review_test)

review_sample_set=make_review_set(sample_review)

l= list(review_train_set)


from sklearn.cross_validation import train_test_split

review_train,review_test=train_test_split(merged_dic,test_size=0.3)

"""Export Data with pickle"""
with open('review_test_set.pickle', 'wb') as handle:
    pickle.dump(review_test_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""Export Data with pickle"""
with open('merged_test.pickle', 'wb') as handle:
    pickle.dump(merged_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
review_test_set=pd.read_pickle('review_test_set.pickle')
review_train_set=pd.read_pickle('review_train_set.pickle')
review_sample_set=pd.read_pickle('review_sample_set.pickle')



calculate_average_reviews_per_day(review_sample_set)

calculate_average_word_count(review_test_set)


get_top_mentions(review_set)


review_sample_set['overall']

#Convert set to List
test_list = list(review_sample_set)

#Flat List
test_list = [" ".join(item) for Review in test_list for item in Review.text_cleaned]

flat_list = [item for sublist in test_list for item in sublist]
#Merge List String


test_list_merge = ' '.join([word for word in review_sample_list])