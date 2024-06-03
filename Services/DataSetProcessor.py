from Services.JSONHandler import JSONHandler
from Services.TextProcessor import TextProcessor
from Services.DataBase import DataBase

def process_dataset_without_embeddings(dataset_name, dataset_path, index_path, tfidf_file, processed_docs, feature_file, vectorizer_file, doc_ids_file):

    dataset = JSONHandler.convert_from_json(dataset_path)

    TextProcessor.process_documents(dataset, index_path, tfidf_file, processed_docs, feature_file, vectorizer_file, doc_ids_file)
    

def process_dataset_with_embeddings(dataset_name, dataset_path,embeddings_file,collection_name,ft_model):

    dataset = JSONHandler.convert_from_json(dataset_path)
    TextProcessor.process_documents_with_ft_model(dataset,embeddings_file,collection_name,ft_model)


antique_paths = {
        'dataset_name': 'antique',
        'dataset_path': './DataSets/Antique/JSON/Antique_Docs.json',
        'index_path': './DataSets/Antique/inverted-index.json',
        'tfidf_file': './DataSets/Antique/tfidf-matrix.npz',
        'processed_docs': './DataSets/Antique/processed_docs.json',
        'feature_file': './DataSets/Antique/features.json',
        'vectorizer_file': './DataSets/Antique/vectorizer.pkl',
        'doc_ids_file': './DataSets/Antique/doc_ids.json',
        'embeddings_file': './DataSets/Antique/embeddings.json',
        'collection_name': 'antique',
        'model_path': './DataSets/Models/antique_model'
    }

antique_paths_test = {
        'dataset_name': 'antique',
        'dataset_path': './DataSets/Antique/JSON/First_1000_Docs.json',
        'index_path': './DataSets/Antique/TEST/inverted-index.json',
        'tfidf_file': './DataSets/Antique/TEST/tfidf-matrix.npz',
        'processed_docs': './DataSets/Antique/TEST/processed_docs.json',
        'feature_file': './DataSets/Antique/TEST/features.json',
        'vectorizer_file': './DataSets/Antique/TEST/vectorizer.pkl',
        'doc_ids_file': './DataSets/Antique/TEST/doc_ids.json',
        'embeddings_file': './DataSets/Antique/TEST/embeddings.json',
        'collection_name': 'antique_test'
    }

quora_paths = {
        'dataset_name': 'quora',
        'dataset_path': './DataSets/Quora/JSON/quora-test-docs.json',
        'index_path': './DataSets/Quora/inverted-index.json',
        'tfidf_file': './DataSets/Quora/tfidf-matrix.npz',
        'processed_docs': './DataSets/Quora/processed_docs.json',
        'feature_file': './DataSets/Quora/features.json',
        'vectorizer_file': './DataSets/Quora/vectorizer.pkl',
        'doc_ids_file': './DataSets/Quora/doc_ids.json',
        'embeddings_file': './DataSets/Quora/embeddings.json',
        'collection_name': 'quora'
    }


def get_data_set_files(dataset):
    if(dataset=='antique'):
        return antique_paths
    else:
        return quora_paths

def process_datasets(data_set, embedding,ft_model=None):
    if data_set == 'antique':
        if embedding:
            process_dataset_with_embeddings(antique_paths["dataset_name"],antique_paths["dataset_path"],antique_paths["embeddings_file"],antique_paths["collection_name"],ft_model)
        else:
            process_dataset_without_embeddings(antique_paths["dataset_name"],antique_paths["dataset_path"],antique_paths["index_path"],antique_paths["tfidf_file"],antique_paths["processed_docs"],antique_paths["feature_file"],antique_paths["vectorizer_file"],antique_paths["doc_ids_file"])
    elif data_set == 'quora':
        if embedding:
            process_dataset_with_embeddings(quora_paths["dataset_name"],quora_paths["dataset_path"],quora_paths["embeddings_file"],quora_paths["collection_name"],ft_model)
        else:
            process_dataset_without_embeddings(quora_paths["dataset_name"],quora_paths["dataset_path"],quora_paths["index_path"],quora_paths["tfidf_file"],quora_paths["processed_docs"],quora_paths["feature_file"],quora_paths["vectorizer_file"],quora_paths["doc_ids_file"])
    elif data_set == 'antique_test': 
        if embedding:
            process_dataset_with_embeddings(antique_paths_test["dataset_name"],antique_paths_test["dataset_path"],antique_paths_test["embeddings_file"],antique_paths_test["collection_name"],ft_model)
        else:
            process_dataset_without_embeddings(antique_paths_test["dataset_name"],antique_paths_test["dataset_path"],antique_paths_test["index_path"],antique_paths_test["tfidf_file"],antique_paths_test["processed_docs"],antique_paths_test["feature_file"],antique_paths_test["vectorizer_file"],antique_paths_test["doc_ids_file"])