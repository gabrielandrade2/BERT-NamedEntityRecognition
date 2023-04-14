import mojimoji
import pandas as pd
from rapidfuzz import fuzz, process

from util import text_utils
from util.text_utils import EntityNormalizer, DrugNameMatcher


class TranslationNotFound(BaseException):
    pass


class HyakuyakuList:

    def __init__(self, path='data/HYAKUYAKU_FULL_v20210706.xlsx'):
        self.df = pd.read_csv(path)

    def get_surface_forms(self):
        return set(self.df['出現形'].dropna())

    def get_general_names(self):
        return set(self.df['一般名'].dropna())

    def get_general_name(self, term):
        return self.df[self.df['出現形'] == term]['一般名'].values[0]

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

    def __init__(self, hyakuyaku_list, matching_method=fuzz.ratio, matching_threshold=0):
        self.list = hyakuyaku_list
        self.candidates = {mojimoji.han_to_zen(x) for x in self.list.get_surface_forms()}
        self.matching_method = matching_method
        self.matching_threshold = matching_threshold

    def normalize(self, term):
        term = mojimoji.han_to_zen(term)
        preferred_candidate = process.extractOne(term, self.candidates, scorer=self.matching_method)
        score = preferred_candidate[1]

        if score > self.matching_threshold:
            ret = self.list.get_general_name(preferred_candidate[0])
            if pd.isna(ret):
                return '', score
            return ret, score
        else:
            return '', score

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
