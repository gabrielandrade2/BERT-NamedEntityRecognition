import re
from abc import abstractmethod, ABCMeta


def preprocessing(texts, remove_core_tag=True):
    """Preprocessing steps for strings.
    Strip strings, remove <core> tags (for now).

    :param texts: List of strings to be processed.
    :param remove_core_tag: Should remove core tag from the texts?
    :return: The list of processed texts.
    """

    processed_texts = list()
    for text in texts:
        if remove_core_tag:
            # Remove all <core> and </core> for now
            text = text.replace('<core>', '')
            text = text.replace('</core>', '')

        # Remove all \n from the beginning and end of the sentences
        text = text.strip()
        processed_texts.append(text)
    return processed_texts


def split_sentences(texts, return_flat_list=True):
    """Given a list of strings, split them into sentences and join everything together into a flat list containing all
     the sentences.

    :param texts: List of strings to be processed.
    :param return_flat_list: If True return a flat list with all the sentences, otherwise a list of lists.
    :return: The list of split sentences.
    """
    processed_texts = list()
    for text in texts:
        processed_text = re.split(
            "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?!])\s\n*|(?<=[^A-zＡ-ｚ0-9０-９ ].)(?<=[。．.?？!！])(?![\.」])\n*", text)
        # processed_text = re.split("(? <=[。?？!！])")  # In case only a simple regex is necessary
        processed_text = [x.strip() for x in processed_text]
        processed_text = [x for x in processed_text if x != '']
        if return_flat_list:
            processed_texts.extend(processed_text)
        else:
            processed_texts.append(processed_text)
    return processed_texts


class EntityNormalizerInterface(metaclass=ABCMeta):

    @abstractmethod
    def normalize(self, term, matching_method):
        pass

    @abstractmethod
    def normalize_list(self, terms, matching_method):
        pass


class DrugNameMatcher(metaclass=ABCMeta):

    @abstractmethod
    def match(self, text, matching_method):
        pass

    @staticmethod
    def exact_match(text1, text2):
        if text1 == text2:
            return 100
        return 0
