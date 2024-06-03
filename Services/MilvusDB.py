from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
import re
import numpy as np
class MilvusDB:
    @staticmethod
    def connect(host="localhost", port="19530"):
        connections.connect("default", host=host, port=port)
        print("Connected to Milvus at {}:{}".format(host, port))

    @staticmethod
    def create_collection(collection_name, dim, field_name="vector", index_type="IVF_FLAT", metric_type="COSINE", index_params={"nlist": 16384}):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name=field_name, dtype=DataType.FLOAT_VECTOR, dim=dim, metric_type=metric_type)
        ]
        schema = CollectionSchema(fields, description="Document Vectors")
        collection = Collection(name=collection_name, schema=schema)
        

        index = {
            "field_name": field_name,
            "index_type": index_type,
            "metric_type": metric_type,  
            "params": index_params
        }
        collection.create_index(field_name, index)
        print(f"Index created on {field_name} with type {index_type} and metric {metric_type}")

        return collection

    @staticmethod
    def get_collection(collection_name):
        try:
            return Collection(collection_name)
        except Exception as e:
            print(f"Error accessing collection: {e}")
            return None
        

    @staticmethod
    def drop_all_collections():
        try:

            collections = utility.list_collections()

            for collection_name in collections:
                collection = Collection(name=collection_name)
                collection.drop()
                print(f"Dropped collection: {collection_name}")
        except Exception as e:
            print(f"Failed to drop collections: {e}")


    @staticmethod
    def get_data_by_doc_id(collection_name, doc_id):
        collection = MilvusDB.get_collection(collection_name)
        if collection is None:
            print("Collection does not exist.")
            return None
        try:

            collection.load()

            query = f"doc_id == '{doc_id}'"

            results = collection.query(expr=query, output_fields=["vector"])
            return results
        except Exception as e:
            print(f"Failed to retrieve data: {e}")
            return None
        

    @staticmethod
    def search_similar_vectors(collection_name, query_vector, top_k=10, search_params={"metric_type": "COSINE", "params": {"ef": 100}}):
        collection = MilvusDB.get_collection(collection_name)
        if collection is None:
            print("Collection does not exist.")
            return None
        try:
            collection.load()

            # Normalize the query vector if using COSINE metric
            if search_params.get("metric_type", "").upper() == "COSINE":
                norm = np.linalg.norm(query_vector)
                if norm > 0:
                    query_vector = [x / norm for x in query_vector]

            search_results = collection.search(
                data=[query_vector], 
                anns_field="vector", 
                param=search_params, 
                limit=top_k, 
                output_fields=["doc_id"]
            )

            return str(search_results)
        except Exception as e:
            print(f"Failed to search for similar vectors: {e}")
            return None

    @staticmethod
    def extract_doc_ids(results):
        doc_ids = []
        if results is not None:
            # Find all entries that look like dictionaries
            entries = re.findall(r'\{.*?\}', results)
            # Extract doc_ids using regex adjusted for escaped single quotes and backslashes
            for entry in entries:
                match = re.search(r"\\'doc_id\\': \\'(.*?)\\'", entry)
                if match:
                    doc_ids.append(match.group(1))
        return doc_ids