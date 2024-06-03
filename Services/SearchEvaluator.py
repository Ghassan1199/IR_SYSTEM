from Services.JSONHandler import JSONHandler
import math

class SearchEvaluator:

    @staticmethod
    def load_qrels(file_path, relevance_threshold=0):
        qrels = {}
        try:
            data = JSONHandler.convert_from_json(file_path)
            for entry in data:
                query_id = entry.get("query_id")
                doc_id = entry.get("doc_id")
                relevance = entry.get("relevance", 0)
                if relevance > relevance_threshold:
                    if query_id not in qrels:
                        qrels[query_id] = set()
                    qrels[query_id].add(doc_id)
        except Exception as e:
            print(f"Failed to load or parse the qrels file: {e}")  # Log or handle the exception appropriately
            raise
        return qrels
    

    @staticmethod
    def calculate_mrr(queries_results, qrels):
            mrr_total = 0
            num_queries = len(queries_results)
            for query_id, retrieved_docs in queries_results.items():
                relevant_docs = qrels.get(query_id, set())

                reciprocal_rank = 0
                for rank, doc_id in enumerate(retrieved_docs, start=1):
                    if doc_id in relevant_docs:
                        reciprocal_rank = 1 / rank
                        break
                mrr_total += reciprocal_rank

            return mrr_total / num_queries if num_queries > 0 else 0

    @staticmethod
    def calculate_map(retrieved_docs, relevant_docs):
        if not relevant_docs:
            return 0
        hit_count = 0
        sum_precisions = 0
        for i, doc in enumerate(retrieved_docs):  # Limit to top 10 documents
            if doc in relevant_docs:
                hit_count += 1
                precision_at_i = hit_count / (i + 1)
                sum_precisions += precision_at_i
        return sum_precisions / len(relevant_docs)


    @staticmethod
    def calculate_precision(retrieved_docs, relevant_docs):
        if not retrieved_docs:
            return 0
        top_10_retrieved_docs = retrieved_docs[:10]
        retrieved_relevant_docs = set(top_10_retrieved_docs) & set(relevant_docs)
        return len(retrieved_relevant_docs) / len(top_10_retrieved_docs)

    @staticmethod
    def calculate_recall(retrieved_docs, relevant_docs):
        if not relevant_docs:
            return 0
        retrieved_relevant_docs = set(retrieved_docs) & set(relevant_docs)
        return len(retrieved_relevant_docs) / len(relevant_docs)

    @staticmethod
    def calculate_f1_score(precision, recall):
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def evaluate_search_engine(query_id, search_results, qrels):
        retrieved_docs = [doc['doc_id'] for doc in search_results if 'doc_id' in doc]
        relevant_docs = qrels.get(query_id, set())
        queries_results = {query_id: retrieved_docs}
        precision = SearchEvaluator.calculate_precision(retrieved_docs, relevant_docs)
        recall = SearchEvaluator.calculate_recall(retrieved_docs, relevant_docs)

        average_precision = SearchEvaluator.calculate_map(retrieved_docs, relevant_docs)
        mrr = SearchEvaluator.calculate_mrr(queries_results, qrels)

        return {

            "P@10": precision,
            "R": recall,
            "MAP": average_precision,
            "MRR" : mrr
        }

    @staticmethod
    def evaluate_search_engine_with_embedding(query_id, retrieved_docs, qrels):
        relevant_docs = qrels.get(query_id, set())
        queries_results = {query_id: retrieved_docs}
        precision = SearchEvaluator.calculate_precision(retrieved_docs, relevant_docs)
        recall = SearchEvaluator.calculate_recall(retrieved_docs, relevant_docs)

        map = SearchEvaluator.calculate_map(retrieved_docs, relevant_docs)
        mrr = SearchEvaluator.calculate_mrr(queries_results, qrels)

        return {
            "P@10": precision,
            "R": recall,
            "MAP": map,
            "MRR": mrr
        }
        