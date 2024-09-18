import streamlit as st
from sklearn.decomposition import PCA
import pandas as pd 
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from utilstest import read_pdf, split_document, classify_and_cluster, get_cluster_sentiments

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
            cluster_sentiments = get_cluster_sentiments(doc_content_list, clusters, n_clusters)
            
            st.header("Clusters and Topics")
            for i, cluster_topics in enumerate(topics):
                if st.button(f"Explore Cluster {i+1}"):
                    st.write(f"Cluster {i+1}:")
                    for topic in cluster_topics:
                        st.write(topic)
            
            st.header("Cluster Word Clouds")
            create_wordcloud(tfidf_matrix, clusters, vectorizer, n_clusters)
            
            st.header("Cluster Sentiments")
            for i, (polarity, subjectivity) in enumerate(cluster_sentiments):
                st.write(f"Cluster {i+1}: Polarity = {polarity:.2f}, Subjectivity = {subjectivity:.2f}")
        else:
            st.write("Document is too small to cluster.")
