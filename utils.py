# import re
# import nltk
# from nltk.tokenize import sent_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.decomposition import LatentDirichletAllocation as LDA

# from PyPDF2 import PdfReader
# import pandas as pd
# import numpy as np
# nltk.download('punkt_tab')

# # Helper function to read PDF file
# def read_pdf(file):
#     reader = PdfReader(file)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# # Function to split document content into meaningful sections
# def split_document(doc_content):
#     # Split by multiple newlines or multiple spaces (for paragraphs)
#     paragraphs = re.split(r'\n{2,}|\s{2,}', doc_content)

#     # Tokenize each paragraph into sentences
#     sentences = []
#     for paragraph in paragraphs:
#         sentences.extend(sent_tokenize(paragraph))

#     # Filter out empty sentences or very short sentences
#     sentences = [sentence for sentence in sentences if len(sentence.strip()) > 2]

#     return sentences


# from sklearn.metrics.pairwise import cosine_similarity

# from sklearn.metrics.pairwise import cosine_similarity


# def remove_duplicate_topics(topics, feature_names):
#     unique_topics = []
#     topic_vectors = []

#     # Convert feature names to a list if it's not already
#     if isinstance(feature_names, np.ndarray):
#         feature_names = feature_names.tolist()

#     for topic in topics:
#         # Create a vector representation of the topic
#         topic_vector = [0] * len(feature_names)
#         for word in topic:
#             if word in feature_names:
#                 topic_vector[feature_names.index(word)] = 1
#         topic_vectors.append(topic_vector)

#     filtered_topics = []

#     for i, topic_vector in enumerate(topic_vectors):
#         is_unique = True
#         for unique_vector in topic_vectors[:i]:
#             if cosine_similarity([topic_vector], [unique_vector])[0][0] > 0.9:
#                 is_unique = False
#                 break
#         if is_unique:
#             filtered_topics.append(topics[i])

#     return filtered_topics


# def topic_modeling(tfidf_matrix, feature_names, n_topics=5):
#     lda = LDA(n_components=n_topics, random_state=0)
#     lda.fit(tfidf_matrix)
#     topics = []
#     for idx, topic in enumerate(lda.components_):
#         topic_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
#         topics.append(topic_words)
#     return topics


# # # Function to classify and cluster document content
# # def classify_and_cluster(doc_content_list, n_clusters=5):
# #     # Preprocess document content using TF-IDF
# #     vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
# #     tfidf_matrix = vectorizer.fit_transform(doc_content_list)

# #     # KMeans clustering
# #     kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
# #     clusters = kmeans.fit_predict(tfidf_matrix)

# #     # Topic Modeling for each cluster
# #     topics = []
# #     for cluster_num in range(n_clusters):
# #         cluster_indices = [i for i, label in enumerate(clusters) if label == cluster_num]
# #         cluster_tfidf = tfidf_matrix[cluster_indices]
# #         topics.append(topic_modeling(cluster_tfidf, vectorizer.get_feature_names_out()))

# #     return tfidf_matrix, clusters, topics


# def classify_and_cluster(doc_content_list, n_clusters=5):
#     vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
#     tfidf_matrix = vectorizer.fit_transform(doc_content_list)

#     # KMeans clustering
#     kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
#     clusters = kmeans.fit_predict(tfidf_matrix)

#     # Get topics for each cluster using topic modeling
#     topics = []
#     feature_names = vectorizer.get_feature_names_out()  # Ensure this is a list of strings

#     for cluster_num in range(n_clusters):
#         cluster_indices = [i for i, label in enumerate(clusters) if label == cluster_num]

#         # If a cluster has data points, generate topics for it
#         if len(cluster_indices) > 0:
#             cluster_tfidf = tfidf_matrix[cluster_indices]
#             cluster_topics = topic_modeling(cluster_tfidf, feature_names)
#             filtered_topics = remove_duplicate_topics(cluster_topics, feature_names)
#             topics.append(filtered_topics)
#         else:
#             topics.append([])  # Empty list for clusters with no data points

#     return tfidf_matrix, clusters, topics, vectorizer




import re
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.metrics.pairwise import cosine_similarity
# from transformers import pipeline

from PyPDF2 import PdfReader
import pandas as pd
import numpy as np
nltk.download('punkt')

# Helper function to read PDF file
def read_pdf(file):
    reader = PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to split document content into meaningful sections
def split_document(doc_content):
    paragraphs = re.split(r'\n{2,}|\s{2,}', doc_content)
    sentences = []
    for paragraph in paragraphs:
        sentences.extend(sent_tokenize(paragraph))
    sentences = [sentence for sentence in sentences if len(sentence.strip()) > 2]
    return sentences

def remove_duplicate_topics(topics, feature_names):
    unique_topics = []
    topic_vectors = []
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()
    for topic in topics:
        topic_vector = [0] * len(feature_names)
        for word in topic:
            if word in feature_names:
                topic_vector[feature_names.index(word)] = 1
        topic_vectors.append(topic_vector)
    filtered_topics = []
    for i, topic_vector in enumerate(topic_vectors):
        is_unique = True
        for unique_vector in topic_vectors[:i]:
            if cosine_similarity([topic_vector], [unique_vector])[0][0] > 0.9:
                is_unique = False
                break
        if is_unique:
            filtered_topics.append(topics[i])
    return filtered_topics

def topic_modeling(tfidf_matrix, feature_names, n_topics=5):
    lda = LDA(n_components=n_topics, random_state=0)
    lda.fit(tfidf_matrix)
    topics = []
    for idx, topic in enumerate(lda.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics.append(topic_words)
    return topics

def classify_and_cluster(doc_content_list, n_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
    tfidf_matrix = vectorizer.fit_transform(doc_content_list)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(tfidf_matrix)
    topics = []
    feature_names = vectorizer.get_feature_names_out()
    for cluster_num in range(n_clusters):
        cluster_indices = [i for i, label in enumerate(clusters) if label == cluster_num]
        if len(cluster_indices) > 0:
            cluster_tfidf = tfidf_matrix[cluster_indices]
            cluster_topics = topic_modeling(cluster_tfidf, feature_names)
            filtered_topics = remove_duplicate_topics(cluster_topics, feature_names)
            topics.append(filtered_topics)
        else:
            topics.append([])
    return tfidf_matrix, clusters, topics, vectorizer

def sentiment_analysis(doc_content_list):
    sentiment_pipeline = pipeline("sentiment-analysis")
    sentiments = sentiment_pipeline(doc_content_list)
    return sentiments

