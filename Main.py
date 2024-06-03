from flask import Flask , jsonify , request
from Services.SearchEngine import SearchEngine
from Services.DataSetProcessor import get_data_set_files
import json
from bson import ObjectId

app = Flask(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


@app.route('/<dataset>/',methods=['POST'])
def search(dataset):
    data = request.get_json()
    query = data["query"]

    max_number = data.get("max_number", -1)

    paths = get_data_set_files(dataset) 

    result = SearchEngine.search_without_embedding(query, paths["tfidf_file"], dataset, paths["vectorizer_file"], paths["doc_ids_file"],max_number)
    
    # Use the custom JSON encoder

    response = json.dumps(result, cls=CustomJSONEncoder)
    return response


if __name__ == '__main__':

    app.run(debug=True)