import mojimoji
import pandas as pd
from rapidfuzz import fuzz, process

from util.text_utils import EntityNormalizer


class OICIDatabase:

    def __init__(self, path='data/OICI extracted symptoms w_ meddra_humancheck..xlsx'):
        self.df = pd.read_excel(path)


class OICINormalizer(EntityNormalizer):

    def __init__(self, database: OICIDatabase, matching_method=fuzz.ratio, threshold=60):
        self.database = database.df
        self.matching_method = matching_method
        self.threshold = threshold
        self.candidates = {mojimoji.han_to_zen(x) for x in self.database['word']}

    def normalize(self, term):
        temp = []

        exact_match = self.database[self.database['word'] == term]['human_check'].to_list()

        if exact_match:
            if (not pd.isna(exact_match[0])) and str(exact_match[0]) != '-1':
                if not str(exact_match[0]) == '[AMB]':
                    return str(exact_match[0])
                return term
            return ''

        preferred_candidate = process.extractOne(term, self.candidates, scorer=self.matching_method)

        score = preferred_candidate[1]

        if score > self.threshold:
            return self.normalize(preferred_candidate[0])
        else:
            return ''
