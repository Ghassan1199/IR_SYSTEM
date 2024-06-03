from nltk.tokenize import word_tokenize
from Services.TextProcessor import TextProcessor
from nltk.corpus import wordnet
import nltk
from nltk import Tree
import nltk
from nltk import pos_tag

class QueryProcessor:

    @classmethod
    def process_query_with_ft_model(cls, query_text,ft_model):
        return ft_model.get_sentence_vector(query_text).tolist()

    @classmethod
    def process_query(cls, query):
        corrected_query = TextProcessor.correct_sentence_spelling(query)
        return TextProcessor.process_text(corrected_query)

    @classmethod
    def recognize_entities(cls, query):
        tokens = nltk.word_tokenize(query)
        pos_tags = nltk.pos_tag(tokens)
        chunked_ner = nltk.ne_chunk(pos_tags)
        entities = []

        for chunk in chunked_ner:
            if isinstance(chunk, Tree):
                ent_name = ' '.join(c[0] for c in chunk.leaves())  # the word itself
                ent_type = chunk.label()  # the entity type
                entities.append((ent_name, ent_type))
        return entities
    
    @classmethod
    def expand_query_with_synonyms(cls, query):
        tokens = word_tokenize(query)  # Tokenize the query using NLTK
        pos_tags = pos_tag(tokens)  # Get part-of-speech tags for each token

        expanded_query = set(tokens)  # Start with the original tokens

        for token, pos in pos_tags:

            wordnet_pos = cls.nltk_pos_to_wordnet_pos(pos)
            if wordnet_pos:  # Ensure that the POS is supported by WordNet
                synsets = wordnet.synsets(token, wordnet_pos)
                for syn in synsets:
                    if syn:
                        for lemma in syn.lemmas():
                        # Add synonyms, replacing underscores with spaces
                            expanded_query.add(lemma.name().replace('_', ' '))

        return " ".join(expanded_query)

    @staticmethod
    def nltk_pos_to_wordnet_pos(nltk_pos):
        """ Convert NLTK POS tags to WordNet POS tags """
        tag_map = {
            'NN': wordnet.NOUN, 'NNS': wordnet.NOUN, 'NNP': wordnet.NOUN, 'NNPS': wordnet.NOUN,
            'VB': wordnet.VERB, 'VBD': wordnet.VERB, 'VBG': wordnet.VERB, 'VBN': wordnet.VERB, 'VBP': wordnet.VERB, 'VBZ': wordnet.VERB,
            'JJ': wordnet.ADJ, 'JJR': wordnet.ADJ, 'JJS': wordnet.ADJ,
            'RB': wordnet.ADV, 'RBR': wordnet.ADV, 'RBS': wordnet.ADV
        }
        return tag_map.get(nltk_pos[:2], None)
