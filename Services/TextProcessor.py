import string
import json
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer ,WordNetLemmatizer
from nltk.corpus import wordnet,stopwords
from spellchecker import SpellChecker
from typing import List
from unidecode import unidecode
from collections import defaultdict
from multiprocessing import Pool, Manager
from functools import partial
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import pickle
from tqdm import tqdm
import numpy as np
import re
from decimal import Decimal


# Define update_progress at the module level
def update_progress(pbar, lock):
    with lock:
        pbar[0] += 1

class TextProcessor:
    stemmer = SnowballStemmer(language="english")
    lemmatizer = WordNetLemmatizer()

    # Assuming the vectorizer is a class attribute for reusability
    vectorizer = TfidfVectorizer(
        stop_words='english',
        lowercase=True,
        smooth_idf=True,
        sublinear_tf=True
    )

    @classmethod
    def initialize_progress_lock(cls, progress, lock):
        global shared_progress
        global shared_lock
        shared_progress = progress
        shared_lock = lock

    @classmethod
    def process_document_wrapper(cls, doc, total_docs, update_progress):
        result = cls.process_doc(doc)
        update_progress()  # Update the shared counter
        return result

    @classmethod
    def process_documents(cls, documents, index_file, tfidf_file, processed_docs_file, feature_file_path, vectorizer_file, doc_ids_file):
        manager = Manager()
        pbar = manager.list([0])  # This list will have one element to act as a counter
        lock = manager.Lock()  # Create a Lock using Manager
        total_docs = len(documents)

        # Initialize the progress bar
        tqdm_pbar = tqdm(total=total_docs, desc="Processing Documents")

        with Pool() as pool:
            func = partial(cls.process_document_wrapper, total_docs=total_docs, update_progress=partial(update_progress, pbar, lock))
            for _ in pool.imap_unordered(func, documents):
                tqdm_pbar.update(1)  # Update the progress bar for each document processed
                
            processed_docs = pool.map(func, documents)
        tqdm_pbar.close()  # Close the progress bar after all documents are processed

        cls.save_processed_data(processed_docs, index_file, tfidf_file, processed_docs_file, feature_file_path, vectorizer_file, doc_ids_file)

    @classmethod
    def save_processed_data(cls, processed_docs, index_file, tfidf_file, processed_docs_file, feature_file_path,vectorizer_file,doc_ids_file):
        inverted_index = defaultdict(lambda: defaultdict(int))
        doc_freq = defaultdict(int)

        documents = []
        doc_ids = []  # List to store document IDs
        for doc in processed_docs:

            doc_id = doc['doc_id']
            stemmed_text = doc['text']
            documents.append(' '.join(stemmed_text))
            doc_ids.append(doc_id)  # Collect document IDs
            for word in stemmed_text:
                inverted_index[word][doc_id] += 1
                doc_freq[word] += 1

        if documents:
            # Create TF-IDF matrix using the class vectorizer
            tfidf_matrix = cls.vectorizer.fit_transform(documents)
            with open(vectorizer_file, 'wb') as f:
                pickle.dump(cls.vectorizer, f)
            # Save the TF-IDF matrix to a compressed sparse matrix file
            cls.save_tfidf_matrix(tfidf_matrix, doc_ids, tfidf_file, doc_ids_file)
            # Extract and save features
            cls.extract_features_from_tfidf(tfidf_matrix, feature_file_path)

        try:
            with open(index_file, 'w') as idx_file:
                json.dump(inverted_index, idx_file, indent=4)
            print("Inverted index successfully saved to", index_file)
        except Exception as e:
            print("Failed to save inverted index:", e)

        # Save the processed documents
        try:
            with open(processed_docs_file, 'w') as doc_file:
                json.dump(processed_docs, doc_file, indent=4)
            print("Processed documents successfully saved to", processed_docs_file)
        except Exception as e:
            print("Failed to save processed documents:", e)

    @classmethod
    def process_text(cls, text):
        text_without_links = cls.remove_links(text)
        tokens = word_tokenize(text_without_links)
        removed_accent = cls.remove_accents_from_list(tokens)
        filtered = cls.remove_punctuation_from_list(removed_accent)
        filtered_text = cls.remove_stopwords(filtered)
        removed_markers = cls.remove_markers(filtered_text)
        removed_apostrophe = cls.remove_apostrophe(removed_markers)
        without_under_score = cls.remove_under_score(removed_apostrophe)
        without_dashes = cls.remove_dashes(without_under_score)
        lemmatized = cls.lemmatize_text(without_dashes)
        stemmed = cls.stem_words(lemmatized)
        lower_text =cls.to_lower(stemmed)
        return lower_text

    @classmethod
    def process_text_for_embedding(cls, text):
        text_without_links = cls.remove_links(text)
        tokens = word_tokenize(text_without_links)
        removed_accent = cls.remove_accents_from_list(tokens)
        filtered = cls.remove_punctuation_from_list(removed_accent)
        filtered_text = cls.remove_stopwords(filtered)
        removed_markers = cls.remove_markers(filtered_text)
        removed_apostrophe = cls.remove_apostrophe(removed_markers)
        without_under_score = cls.remove_under_score(removed_apostrophe)
        without_dashes = cls.remove_dashes(without_under_score)
        lower_text =cls.to_lower(without_dashes)
        return lower_text
    
    

    @classmethod
    def remove_links(cls, text):
        url_pattern = r'https?://[^\s,]+|www\.[^\s,]+'
        cleaned_text = re.sub(url_pattern, '', text)
        cleaned_text = re.sub(r'\b(?:[A-Za-z0-9]+\.)+[A-Za-z0-9]+(?:\s|\b)', '', cleaned_text)
        return cleaned_text
    
    @classmethod
    def remove_dashes(cls, tokens):
        return [token.replace('-', '') for token in tokens]


    @classmethod
    def remove_apostrophe(cls,tokens: List[str]) -> List[str]:
        new_tokens = []
        for token in tokens:
            new_tokens.append(str(np.char.replace(token, "'", " ")))
        return new_tokens
    
    @staticmethod
    def remove_under_score(tokens: List[str]) -> List[str]:
        new_tokens = []
        for token in tokens:
            new_tokens.append(re.sub(r'_', ' ', token))
        return new_tokens
    
    @classmethod
    def remove_markers(cls,tokens: List[str]) -> List[str]:
        new_tokens = []
        for token in tokens:
            new_tokens.append(re.sub(r'\u00AE', '', token))
        return new_tokens
    
    
    @classmethod
    def to_lower(cls,tokens: List[str]) -> List[str]:
        lower_tokens = []
        for token in tokens:
            lower_token = str(np.char.lower(token))
            lower_tokens.append(lower_token)
        return lower_tokens


    
    @classmethod
    def remove_stopwords(cls, tokens):
        stop_words = set(stopwords.words('english'))
        filtered_text = [word for word in word_tokenize(tokens) if word.lower() not in stop_words]
        return filtered_text

    @classmethod
    def lemmatize_text(cls, tokens):
        # Ensure tokens are in list form and tokenize if input is a string
        tokens = word_tokenize(tokens) if isinstance(tokens, str) else tokens
        # Get POS tags for each token
        pos_tags = pos_tag(tokens)

        lemmatized_words = []
        for token, pos in pos_tags:
            # Convert the POS tag to a format recognized by WordNetLemmatizer
            wordnet_pos = cls.get_wordnet_pos(pos)
            # Lemmatize the token with the appropriate POS tag
            if wordnet_pos:
                lemma = cls.lemmatizer.lemmatize(token, wordnet_pos)
            else:
                lemma = cls.lemmatizer.lemmatize(token)
            lemmatized_words.append(lemma)

        return lemmatized_words

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        """Convert treebank POS tags to a WordNetLemmatizer compatible format"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None
    
    @classmethod
    def normalize_abbreviations(cls, tokens: List[str]) -> List[str]:
        resolved_terms = {}
        for token in tokens:
            if len(token) >= 2:
                synsets = wordnet.synsets(token)
                if synsets and synsets[0]:  # Ensure synsets is not empty and the first element is not None
                    lemmas = synsets[0].lemmas()
                    if lemmas:  # Check if lemmas list is not empty
                        resolved_term = lemmas[0].name()  # Safely access the first lemma's name
                        resolved_terms[token] = resolved_term
        # Efficiently replace tokens using list comprehension and dictionary get method
        return [resolved_terms.get(token, token) for token in tokens]

    @classmethod
    def remove_punctuation_from_list(cls, tokens):
        table = str.maketrans('', '', string.punctuation)
        filtered_words = [word.translate(table) for word in tokens]
        filtered_words = ' '.join([word for word in filtered_words if word])  # Join words into a single string
        return filtered_words

    @classmethod
    def remove_accents_from_list(cls, tokens):
        return [unidecode(word) for word in tokens]

    @classmethod
    def correct_sentence_spelling(cls, text: str) -> str:
        spell = SpellChecker()
        tokens = text.split()  # Tokenize the input string by spaces
        misspelled = spell.unknown(tokens)  # Find misspelled words

        for i, token in enumerate(tokens):
            if token in misspelled:
                corrected = spell.correction(token)  # Get the corrected spelling
                if corrected:
                    tokens[i] = corrected  # Replace the token with its corrected form

        corrected_text = ' '.join(tokens)  # Join the corrected tokens back into a single string
        return corrected_text

    @classmethod
    def stem_words(cls, tokens):
        return [cls.stemmer.stem(word) for word in tokens]

    @classmethod
    def build_inverted_index(cls, stemmed_text, doc_id, inverted_index):
        term_freq = defaultdict(int)
        for term in stemmed_text:
            term_freq[term] += 1
        for term, freq in term_freq.items():
            inverted_index[term][doc_id] = freq

    @classmethod
    def create_tfidf_matrix(cls, documents):
        vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            smooth_idf=True,
            sublinear_tf=True,  # Apply sublinear TF scaling
        )
        # Ensure documents are strings
        documents = [' '.join(doc) if isinstance(doc, list) else doc for doc in documents]
        print(documents)
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        return tfidf_matrix, feature_names

    @classmethod
    def save_tfidf_matrix(cls, tfidf_matrix, doc_ids, tfidf_file_path, doc_ids_file_path):
        # Save the TF-IDF matrix
        scipy.sparse.save_npz(tfidf_file_path, tfidf_matrix)
        print(f"TF-IDF matrix successfully saved to {tfidf_file_path}.")

        # Save the document IDs
        with open(doc_ids_file_path, 'w') as f:
            json.dump(doc_ids, f)
        print(f"Document IDs successfully saved to {doc_ids_file_path}.")

    @classmethod
    def load_tfidf_matrix_from_csm(cls, file_path):
        # Load the TF-IDF matrix from compressed sparse row (CSR) format
        return scipy.sparse.load_npz(file_path)

    @classmethod
    def extract_features_from_tfidf(cls, tfidf_matrix, feature_file_path):
        # Extract feature names from the vectorizer
        feature_names = cls.vectorizer.get_feature_names_out()
        # Save to a JSON file
        with open(feature_file_path, 'w') as f:
            json.dump(feature_names.tolist(), f, indent=4)
        print(f"Features successfully saved to {feature_file_path}")

    @classmethod
    def load_tfidf_matrix_and_doc_ids(cls, tfidf_file_path, doc_ids_file_path):
        # Load the TF-IDF matrix
        tfidf_matrix = scipy.sparse.load_npz(tfidf_file_path)

        # Load the document IDs
        with open(doc_ids_file_path, 'r') as f:
            doc_ids = json.load(f)

        return tfidf_matrix, doc_ids

    @classmethod
    def process_doc(cls, doc):
        text = ""
        if isinstance(doc["text"], str):
            text = doc["text"]
        elif hasattr(doc["text"], '__iter__') and not isinstance(doc["text"], (str, bytes)):
            try:
                text = " ".join(str(item) for item in doc["text"])
            except TypeError:
                raise ValueError(f"doc['text'] contains non-string/bytes elements: {doc['text']}")
        else:
            text = str(doc["text"])

        # Process text using the updated process_text method
        processed_text = cls.process_text(text)

        doc_id = doc["doc_id"]

        # Compute term frequency in the document
        term_freq = defaultdict(int)
        for word in processed_text:
            term_freq[word] += 1

        return {"doc_id": doc_id, "text": processed_text, "term_freq": term_freq}
    

    @classmethod
    def process_documents_with_ft_model(cls, documents, embeddings_file, collection_name, ft_model):
        # Ensure the collection exists or create it

        processed_docs = []

        # Initialize tqdm progress bar for the first loop
        pbar = tqdm(total=len(documents), desc="Processing documents with FastText model")
        for doc in documents:
            text = doc.get('text', '')
            processed_text = cls.process_text_with_ft_model(text, ft_model)  
            doc_id = doc.get('doc_id')
            processed_docs.append({'doc_id': doc_id, 'vector': processed_text})
            
            
            pbar.update(1)
        pbar.close()  


        with open(embeddings_file, 'w') as f:
            json.dump(processed_docs, f, indent=4)
            print(f"Processed documents successfully saved to {embeddings_file}")

        return processed_docs


    @classmethod
    def process_text_with_ft_model(cls, text,ft_model):
        text_vector = ft_model.get_sentence_vector(text).tolist()
        return text_vector
    

    @staticmethod
    def normalize_vector(vector):
        # Convert vector to a numpy array for efficient computation
        np_vector = np.array(vector, dtype=float)
        # Compute the norm of the vector
        norm = np.linalg.norm(np_vector)
        if norm == 0:
            return vector  # Return the original vector if norm is zero to avoid division by zero
        # Normalize the vector by its norm
        normalized_vector = np_vector / norm
        # Convert normalized numpy array back to list if necessary
        return normalized_vector.tolist()

