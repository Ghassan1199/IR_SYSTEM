from pymongo import MongoClient

class DataBase:
    def __init__(self, db_uri, db_name):
        self.client = MongoClient(db_uri)
        self.db = self.client[db_name]
        self.collection = self.db['documents']
        self.collection.create_index([('base_id', 1)])

    def get_documents_by_ids(self, doc_ids):
        matched_documents = []
        for doc_id in doc_ids:

            cursor = self.collection.find({"doc_id": doc_id})
            documents = list(cursor)
            matched_documents.extend(documents)
        self.close_connection()
        return matched_documents
        
    def close_connection(self):
        self.client.close()
