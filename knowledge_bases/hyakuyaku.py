import mojimoji
import pandas as pd
from thefuzz import fuzz

from util import text_utils
from util.text_utils import EntityNormalizer, DrugNameMatcher


class TranslationNotFound(BaseException):
    pass


class HyakuyakuList:

    def __init__(self, path='data/HYAKUYAKU_FULL_v20210706.xlsx'):
        self.df = pd.read_excel(path)

    def get_surface_forms(self):
        return set(self.df['出現形'].dropna())

    def get_general_names(self):
        return set(self.df['一般名'].dropna())

    def get_english_translation(self, term):
        idx = None
        for column_name in ['出現形', '一般名']:
            column = self.df[column_name]
            matches = column[column == term]
            if not matches.empty:
                idx = matches.index[0]
                break

        if not idx:
            raise TranslationNotFound
        english_names = self.df['出現形英語（DeepL翻訳）']
        return english_names[idx]

    def append_english_name(self, drug):
        try:
            translation = self.get_english_translation(drug)
            if translation:
                return '{} ({})'.format(drug, translation)
        except TranslationNotFound:
            pass
        return drug

class HyakuyakuNormalizer(EntityNormalizer):

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

    def __init__(self, hyakuyaku_list, matching_method=DrugNameMatcher.exact_match):
        self.list = hyakuyaku_list
        self.matching_method = matching_method
        candidate_list = self.list.get_surface_forms()
        candidate_list = candidate_list.union(self.list.get_general_names())
        # Ignore terms with 2 characters or less
        candidate_list = set(filter(lambda x: len(x) > 2, candidate_list))
        # Order list by longer candidates first
        self.candidate_list = sorted(candidate_list, key=len, reverse=True)

    def match(self, text):
        matches = list()
        text = mojimoji.han_to_zen(text)
        for entry in self.candidate_list:
            matches.extend(self.matching_method(text, entry, ignore=matches))
        return matches


class HyakuyakuDrugIOBMatcher(HyakuyakuDrugMatcher):

    def __init__(self, hyakuyaku_list, iob_tag, matching_method=DrugNameMatcher.exact_match):
        super().__init__(hyakuyaku_list, matching_method=matching_method)
        self.iob_tag = iob_tag

    def match(self, text):
        matches = super().match(text)
        return text_utils.tag_matches(text, matches, self.iob_tag)
