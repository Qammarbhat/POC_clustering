# import streamlit as st
# from sklearn.decomposition import PCA
# import pandas as pd 
# import plotly.express as px
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# from utils import read_pdf, split_document, topic_modeling, classify_and_cluster

# # Create a word cloud for each cluster
# def create_wordcloud(tfidf_matrix, clusters, vectorizer, n_clusters):
#     feature_names = vectorizer.get_feature_names_out()
    
#     for cluster_num in range(n_clusters):
#         cluster_indices = [i for i, label in enumerate(clusters) if label == cluster_num]
#         cluster_tfidf = tfidf_matrix[cluster_indices].toarray().sum(axis=0)
        
#         # Create a dictionary for word frequencies
#         word_freq = {feature_names[i]: cluster_tfidf[i] for i in range(len(feature_names)) if cluster_tfidf[i] > 0}
        
#         # Generate the word cloud
#         wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        
#         # Display the word cloud using matplotlib
#         st.subheader(f"Word Cloud for Cluster {cluster_num + 1}")
#         plt.figure(figsize=(10, 5))
#         plt.imshow(wordcloud, interpolation='bilinear')
#         plt.axis('off')
#         st.pyplot(plt)

# # Streamlit app
# st.title("Document Clustering and Topic Modeling with Word Clouds")

# # File uploader for PDF
# uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

# if uploaded_file is not None:
#     # Read PDF content
#     doc_content = read_pdf(uploaded_file)
    
#     if doc_content:
#         # Split the document into sentences/paragraphs
#         doc_content_list = split_document(doc_content)
        
#         if len(doc_content_list) > 1:
#             # Slider to select number of clusters
#             n_clusters = st.slider("Select number of clusters", min_value=2, max_value=min(10, len(doc_content_list)), value=5)
            
#             # Apply KMeans clustering and topic modeling
#             tfidf_matrix, clusters, topics, vectorizer = classify_and_cluster(doc_content_list, n_clusters)
            
#             st.header("Clusters and Topics")
#             # Display cluster and topic information
#             for i, cluster_topics in enumerate(topics):
#                 st.write(f"Cluster {i+1}:")
#                 for topic in cluster_topics:
#                     st.write(topic)
            
#             st.header("Cluster Word Clouds")
#             # Create and display word clouds for each cluster
#             create_wordcloud(tfidf_matrix, clusters, vectorizer, n_clusters)
#         else:
#             st.write("Document is too small to cluster.")


import streamlit as st
from sklearn.decomposition import PCA
import pandas as pd 
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from utils import read_pdf, split_document, classify_and_cluster, sentiment_analysis

def create_wordcloud(tfidf_matrix, clusters, vectorizer, n_clusters):
    feature_names = vectorizer.get_feature_names_out()
    for cluster_num in range(n_clusters):
        cluster_indices = [i for i, label in enumerate(clusters) if label == cluster_num]
        cluster_tfidf = tfidf_matrix[cluster_indices].toarray().sum(axis=0)
        word_freq = {feature_names[i]: cluster_tfidf[i] for i in range(len(feature_names)) if cluster_tfidf[i] > 0}
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        st.subheader(f"Word Cloud for Cluster {cluster_num + 1}")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

st.title("Document Clustering, Topic Modeling, and Sentiment Analysis")

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_file is not None:
    doc_content = read_pdf(uploaded_file)
    if doc_content:
        doc_content_list = split_document(doc_content)
        if len(doc_content_list) > 1:
            n_clusters = st.slider("Select number of clusters", min_value=2, max_value=min(10, len(doc_content_list)), value=5)
            tfidf_matrix, clusters, topics, vectorizer = classify_and_cluster(doc_content_list, n_clusters)
            sentiments = sentiment_analysis(doc_content_list)
            
            st.header("Clusters and Topics")
            for i, cluster_topics in enumerate(topics):
                st.write(f"Cluster {i+1}:")
                for topic in cluster_topics:
                    st.write(topic)
            
            st.header("Cluster Word Clouds")
            create_wordcloud(tfidf_matrix, clusters, vectorizer, n_clusters)
            
            st.header("Sentiment Analysis")
            for i, sentiment in enumerate(sentiments):
                st.write(f"Sentence: {doc_content_list[i]}")
                st.write(f"Sentiment: {sentiment['label']} (Score: {sentiment['score']:.2f})")
        else:
            st.write("Document is too small to cluster.")
