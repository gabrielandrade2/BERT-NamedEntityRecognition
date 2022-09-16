import pandas as pd
from thefuzz import fuzz

from util.text_utils import EntityNormalizer


class OICIDatabase:

    def __init__(self, path='data/OICI extracted symptoms w_ meddra_humancheck..xlsx'):
        self.df = pd.read_excel(path)


class OICINormalizer(EntityNormalizer):

    def __init__(self, database: OICIDatabase, matching_method=fuzz.token_set_ratio, threshold=60):
        self.database = database.df
        self.matching_method = matching_method
        self.threshold = threshold

    def normalize(self, term):
        temp = []

        exact_match = self.database[self.database['word'] == term]['human_check'].to_list()

        if exact_match:
            if (not pd.isna(exact_match[0])) and str(exact_match[0]) != '-1':
                if not str(exact_match[0]) == '[AMB]':
                    return str(exact_match[0])
                return term
            return ''

        for pf in self.database['word']:
            pf = str(pf)
            score = self.matching_method(term, pf)
            temp.append((score, pf))
            if score == 100:
                break

        preferred_candidate = max(temp)

        score = preferred_candidate[0]

        if score > self.threshold:
            return self.normalize(preferred_candidate[1])
        else:
            return ''
