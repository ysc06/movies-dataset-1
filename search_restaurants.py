import pickle
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import re
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from collections import Counter

# Load Word2Vec model
model_path = '../models/GoogleNews-vectors-negative300.bin'
print("Loading Word2Vec model...")
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Utility Functions
def getwordnet_pos(pos):
    """Get WordNet POS tags for lemmatization."""
    if pos.startswith('N'):
        return wn.NOUN
    elif pos.startswith('V'):
        return wn.VERB
    elif pos.startswith('J'):
        return wn.ADJ
    elif pos.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN

def preprocess_query(query):
    """Tokenize, stem/lemmatize, and remove stopwords."""
    stopwords_english = set(stopwords.words('english'))
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')

    # Tokenize and lemmatize/stem query
    tokens = tokenizer.tokenize(query.lower())
    pos_tags = pos_tag(tokens)
    processed_tokens = [
        lemmatizer.lemmatize(word, getwordnet_pos(pos))
        for word, pos in pos_tags if word not in stopwords_english
    ]
    return processed_tokens

def construct_word2vec_vector(tokens, model):
    """Compute the weighted average Word2Vec vector."""
    vector_size = model.vector_size
    weighted_sum = np.zeros(vector_size)
    total_count = 0
    for token in tokens:
        if token in model:
            weighted_sum += model[token]
            total_count += 1
    return weighted_sum / total_count if total_count > 0 else np.zeros(vector_size)

# Retrieval Functions
def bm25_scoring(tokens, bm25, business_index):
    print("Using BM25 scoring...")
    scores = bm25.get_scores(tokens)
    descending_indices = np.argsort(scores)[::-1]
    return pd.DataFrame([(business_index[i], scores[i]) for i in descending_indices], columns=['business_id', 'sim_score'])

def pln_scoring(tokens, vector_dict):
    print("Using Pivoted Length Normalization scoring...")
    query_vector = Counter(tokens)
    sim_scores = [
        (business_id, sum(query_vector.get(term, 0) * weight for term, weight in doc_vector))
        for business_id, doc_vector in vector_dict.items()
    ]
    return pd.DataFrame(sim_scores, columns=['business_id', 'sim_score'])

def word2vec_scoring(tokens, vector_dict, model):
    print("Using Word2Vec scoring...")
    query_vector = construct_word2vec_vector(tokens, model)
    sim_scores = [
        (business_id, np.dot(query_vector, doc_vector))
        for business_id, doc_vector in vector_dict.items()
    ]
    return pd.DataFrame(sim_scores, columns=['business_id', 'sim_score'])

# Main Search Function
def search_restaurants(query, method="bm25", sim_score_wght=0.8, weighted_avg_sentiment=True):
    print("Loading data...")
    with open("yelp_reviews_preprocess_bm25.pkl", 'rb') as bm25_file:
        bm25, business_index = pickle.load(bm25_file)

    with open("yelp_reviews_doc_vectors_pln.pkl", 'rb') as pln_file:
        pln_vector_dict = pickle.load(pln_file)

    with open("yelp_reviews_doc_vectors_word2vec.pkl", 'rb') as w2v_file:
        word2vec_vector_dict = pickle.load(w2v_file)

    lexicon_sentiment_df = pd.read_csv('../../data/yelp_restaurants_lexicon_sentiment_Phila.csv')
    bert_sentiment_df = pd.read_csv('../../data/yelp_restaurants_bert_sentiment_Phila.csv')
    bert_sentiment_df.columns = ['business_id', 'avg_sentiment_bert', 'weighted_avg_sentiment_bert', 
                                 'negative_review_count_bert', 'neutral_review_count_bert', 'positive_review_count_bert']
    sentiment_df = pd.merge(lexicon_sentiment_df, bert_sentiment_df, on='business_id', how='inner')

    user_reviews_df = pd.read_csv('../../data/yelp_restaurants_Phila_final.csv')

    # Preprocess Query
    query_tokens = preprocess_query(query)

    # Select Retrieval Method
    if method == "bm25":
        sim_scores_df = bm25_scoring(query_tokens, bm25, business_index)
    elif method == "pln":
        sim_scores_df = pln_scoring(query_tokens, pln_vector_dict)
    elif method == "word2vec":
        sim_scores_df = word2vec_scoring(query_tokens, word2vec_vector_dict, word2vec_model)
    else:
        raise ValueError("Invalid method. Choose 'bm25', 'pln', or 'word2vec'.")

    # Normalize Scores
    z_score_scaler_sim = StandardScaler()
    sim_scores_df['norm_sim_score'] = z_score_scaler_sim.fit_transform(sim_scores_df['sim_score'].values.reshape(-1, 1))

    # Normalize Sentiment Scores
    sentiment_cols = ['avg_sentiment_vader', 'avg_sentiment_TextBlob', 'avg_sentiment_sentiwordnet', 'avg_sentiment_bert']
    z_score_scaler_sentiment = StandardScaler()
    sentiment_df[[f'norm_{col}' for col in sentiment_cols]] = z_score_scaler_sentiment.fit_transform(sentiment_df[sentiment_cols])
    sentiment_df['norm_avg_sentiment'] = sentiment_df[[f'norm_{col}' for col in sentiment_cols]].mean(axis=1)

    # Merge Scores with Sentiment Data
    scores_df = pd.merge(sim_scores_df, sentiment_df, on='business_id', how='inner')

    # Calculate Weighted Score
    sentiment_weight = 1 - sim_score_wght
    scores_df['weighted_score'] = (
        sim_score_wght * scores_df['norm_sim_score'] + sentiment_weight * scores_df['norm_avg_sentiment']
    )

    # Rank and Return Top Results
    ranked_scores = scores_df.sort_values(by='weighted_score', ascending=False)
    return ranked_scores.merge(user_reviews_df[['business_id', 'restaurant_name']], on='business_id', how='inner')[['business_id', 'restaurant_name', 'weighted_score']].head(10)

# CLI Execution
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        query = sys.argv[1]
        method = sys.argv[2]
        print(f"Processing query: {query} using {method} method.")
        top_10 = search_restaurants(query, method)
        print("\nTop 10 Restaurants:")
        print(top_10.to_string(index=False))
    else:
        print("Usage: python search_restaurants.py '<query>' <method>")
        print("Method must be 'bm25', 'pln', or 'word2vec'.")
