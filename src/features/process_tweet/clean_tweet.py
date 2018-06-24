from src.features.process_text.clean import clean_text

def clean_tweet(review_train_list):
    """Cleans text of tweet."""
    review_test_list.cleaning_log, review_test_list.text_cleaned = clean_text(review_test_list.reviewText, tokenize=True)