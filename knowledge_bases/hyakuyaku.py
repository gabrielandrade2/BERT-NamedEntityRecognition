import pandas as pd
from thefuzz import fuzz

from util.text_utils import EntityNormalizerInterface, DrugNameMatcher


class HyakuyakuList:

    def __init__(self, path='../data/HYAKUYAKU_FULL_v20210706.xlsx'):
        self.df = pd.read_excel(path)

    def get_surface_forms(self):
        return self.df['出現形'].to_list()

    def get_general_names(self):
        return self.df['一般名'].to_list()


class HyakuyakuNormalizer(EntityNormalizerInterface):

    def __init__(self, hyakuyaku_list):
        self.list = hyakuyaku_list
        self.forms = {
            '出現形': self.list.get_surface_forms,
            '一般名': self.list.get_general_names
        }

    def normalize(self, term, matching_method=fuzz.token_set_ratio, threshold=0, form='出現形'):
        candidates = self.forms[form]()
        preferred_candidate = max([(matching_method(term, candidate), candidate) for candidate in candidates])
        score = preferred_candidate[0]

        if score > threshold:
            normalized_term = preferred_candidate[1]
        else:
            normalized_term = term

        return normalized_term, score


class HyakuyakuDrugMatcher(DrugNameMatcher):

    def __init__(self, hyakuyaku_list):
        self.list = hyakuyaku_list
        candidate_list = self.list.get_surface_forms()
        # Order list by longer candidates first
        candidate_list = sorted(candidate_list, key=len, reverse=True)
        # Ignore terms with 2 characters or less
        self.candidate_list = list(filter(lambda x: len(x) > 2, candidate_list))

    def match(self, text, matching_method=DrugNameMatcher.exact_match):
        matches = list()
        i = 0
        for entry in self.candidate_list:
            matches.extend(matching_method(text, entry, ignore=matches))
        return matches
