import json

class JSONHandler:
    @staticmethod
    def convert_from_json(name_of_file):
        with open(name_of_file, 'r', encoding='utf-8') as f:
            file_from_json = f.read()
            return json.loads(file_from_json)

    @staticmethod
    def convert_to_json_file(temp, name_of_file):
        with open(name_of_file, 'w') as f:
            json.dump(temp, f, indent=4)

    @staticmethod
    def add_to_json(temp, name_of_file):
        with open(name_of_file, 'a') as f:
            file_to_json = json.dumps(temp)
            f.write(file_to_json + '\n')

    @staticmethod
    def convert_text_to_json(input_file_path, output_file_path):
        documents = []
        with open(input_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')  # Assuming tab-separated values
                if len(parts) >= 2:
                    doc_id, text = parts[0], ' '.join(parts[1:])
                    documents.append({'doc_id': doc_id, 'text': text})
        
        JSONHandler.convert_to_json_file(documents, output_file_path)

    @staticmethod
    def make_json_file(json_data,filename):
        with open(filename, 'w') as file:
            file.write(json_data)

