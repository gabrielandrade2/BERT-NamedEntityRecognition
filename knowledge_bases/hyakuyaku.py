import pandas as pd
from thefuzz import fuzz

from util.text_utils import EntityNormalizerInterface, DrugNameMatcher


class HyakuyakuList:

    def __init__(self, path='../data/HYAKUYAKU_FULL_v20210706.xlsx'):
        self.df = pd.read_excel(path)

    def get_surface_forms(self):
        return self.df['出現形'].to_list()

    def get_general_name(self):
        return self.df['一般名'].to_list()


class HyakuyakuNormalizer(EntityNormalizerInterface):

    def __init__(self, hyakuyaku_list):
        self.list = hyakuyaku_list

    def normalize(self, term, matching_method=fuzz.token_set_ratio, form='出現形'):
        pass

    def normalize_list(self, terms, matching_method=fuzz.token_set_ratio, form='出現形'):
        pass


class HyakuyakuDrugMatcher(DrugNameMatcher):

    def __init__(self, hyakuyaku_list):
        self.list = hyakuyaku_list

    def match(self, text, matching_method=DrugNameMatcher.exact_match):
        pass
