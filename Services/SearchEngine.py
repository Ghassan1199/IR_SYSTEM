from Services.QueryProcessor import QueryProcessor
from Services.TextProcessor import TextProcessor
import json
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from Services.DataBase import DataBase
from Services.MilvusDB import MilvusDB
import numpy as np


class SearchEngine:
    documents_dict = None  # Class variable to store documents
    documents_cache = {}


    @classmethod
    def load_documents(cls, file_path: str) -> dict:
        if cls.documents_dict is None:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents = json.load(file)
                cls.documents_dict = {doc['doc_id']: doc for doc in documents}
            except Exception as e:
                raise
        return cls.documents_dict
    

    @classmethod
    def load_vectorizer(cls, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def search_with_embedding(cls,query,data_set_name,ft_model,max_number):

        search_results = MilvusDB.search_similar_vectors(data_set_name,ft_model.get_sentence_vector(query).tolist(),max_number)
        doc_ids = MilvusDB.extract_doc_ids(search_results)
        return doc_ids


    @classmethod
    def search_without_embedding(cls, query, tfidf_matrix, data_set_name,vectorizer,doc_ids,min_score,max_number=-1):
        ranked_doc_ids = cls.search_documents_without_embedding(query, tfidf_matrix, doc_ids,vectorizer,max_number,min_score)
        
        database = DataBase('mongodb://localhost:27017/', data_set_name)

        return database.get_documents_by_ids(ranked_doc_ids)
    
    
    @classmethod
    def unified_search(cls, query, data_set_name, tfidf_file_path, vectorizer_file, doc_ids_file_path, ft_model, min_score, max_number=100, tfidf_weight=0.5, embedding_weight=0.5):
        # Load TF-IDF matrix and document IDs
        tfidf_matrix, doc_ids = TextProcessor.load_tfidf_matrix_and_doc_ids(tfidf_file_path, doc_ids_file_path)
        vectorizer = cls.load_vectorizer(vectorizer_file)
    
        # Process query for TF-IDF
        processed_query = QueryProcessor.process_query(query)
        query_vector_tfidf = cls.create_query_vector(processed_query, vectorizer)
    
        # Rank documents based on TF-IDF
        ranked_docs_tfidf = cls.rank_documents(query_vector_tfidf, tfidf_matrix, doc_ids, max_number, min_score)
    
        # Search with embeddings
        search_results_embeddings = MilvusDB.search_similar_vectors(data_set_name, ft_model.get_sentence_vector(query).tolist(), max_number)
        doc_ids_embeddings = MilvusDB.extract_doc_ids(search_results_embeddings)
    
        # Create a dictionary to hold combined scores
        combined_scores = {}
    
        # Update combined scores with weighted TF-IDF scores
        for doc_id, score in ranked_docs_tfidf:
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + score * tfidf_weight
    
        # Update combined scores with weighted embedding scores
        for doc_id, score in doc_ids_embeddings:
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + score * embedding_weight
    
        # Sort documents by combined scores in descending order
        sorted_doc_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)
    
        # If max_number is specified, slice the sorted list to get the top results
        if max_number > 0:
            sorted_doc_ids = sorted_doc_ids[:max_number]
    
        # Fetch documents from the database
        database = DataBase('mongodb://localhost:27017/', data_set_name)
        documents = database.get_documents_by_ids(sorted_doc_ids)

        return documents

    @classmethod
    def load_tfidf_matrix(cls, file_path):
        # Load the TF-IDF matrix from a compressed sparse matrix file
        return TextProcessor.load_tfidf_matrix_from_csm(file_path)

    @classmethod
    def search_documents_without_embedding(cls, query, tfidf_matrix, doc_ids,Vectorizer,max_number,min_score):
        processed_query = QueryProcessor.process_query(query)

        query_vector = cls.create_query_vector(processed_query, Vectorizer)
        ranked_docs = cls.rank_documents(query_vector, tfidf_matrix, doc_ids,max_number,min_score)

        # Unpack the document ID and ignore the similarity score
        return [doc_id for doc_id, _ in ranked_docs]
    
    @classmethod
    def rank_documents(cls, query_vector, tfidf_matrix, doc_ids, max_number, min_score):

        # Calculate cosine similarities between the query vector and all document vectors
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Filter document indices based on the similarity score threshold
        filtered_doc_indices = [(i, score) for i, score in enumerate(cosine_similarities) if score >= min_score]

        # Sort the filtered documents by similarity score in descending order
        sorted_filtered_docs = sorted(filtered_doc_indices, key=lambda x: x[1], reverse=True)

        # Slice the list to get only the top 'max_number' documents
        if(max_number<0):
            top_docs = sorted_filtered_docs
        else:
            top_docs = sorted_filtered_docs[:max_number]


        # Return a list of tuples (doc_id, similarity_score) for the top documents
        return [(doc_ids[idx], score) for idx, score in top_docs]

    @classmethod
    def create_query_vector(cls, processed_query, vectorizer):
        # Transform the processed query using the already fitted vectorizer
        query_vector = vectorizer.transform([' '.join(processed_query)])
        return query_vector