import mojimoji
import pandas as pd
from rapidfuzz import fuzz, process

from util.text_utils import EntityNormalizer


class ManbyoDict:

    def __init__(self, path='data/DIC_C_MANBYO_作業_20220331_v1.0.csv'):
        self.df = pd.read_csv(path)

    # Available fields
    # SEQ
    # ID
    # 出現形
    # 出現形よみ1
    # 出現形よみ2
    # 標準病名
    # 標準病名よみ
    # 症状フラグ
    # 診断名フラグ
    # ICDコード
    # 信頼度レベル
    # 頻度レベル
    # 出現形英語（DeepL翻訳）    共起症状（標準病名）    共起診断（標準病名）    MedisICD10_最新
    # Negativeフラグ

    def getTermList(self):
        return set(self.df['出現形'])

    def getTerm(self, term):
        return self.df[self.df['出現形'] == term].to_dict("records")

    def getTermByEnglishName(self, term):
        return self.df[self.df['出現形英語（DeepL翻訳）'] == term].to_dict("records")

    def getTermICDCode(self, term):
        return self.df[self.df['出現形'] == term]['ICDコード'].item()

    def getTermMedDRA(self, term):
        return self.df[self.df['出現形'] == term]['MedDRA/J (Ver.22): PT/LLT'].item()

    def searchTerm(self, search, num_candidates=1, return_scores=False):
        temp = [(i, fuzz.token_set_ratio(i, search)) for i in self.getTermList()]
        temp = sorted(temp, key=lambda x: x[1], reverse=True)[:num_candidates]
        if not return_scores:
            temp = list(map(lambda x: x[0], temp))
        return temp


class ManbyoNormalizer(EntityNormalizer):

    def __init__(self, database: ManbyoDict, matching_method=fuzz.ratio, matching_threshold=0):
        self.database = database
        self.matching_method = matching_method
        self.matching_threshold = matching_threshold
        self.candidates = {mojimoji.han_to_zen(x) for x in self.database.getTermList()}

    def convert_term(self, term):
        return term

    def normalize(self, term):
        term = mojimoji.han_to_zen(term)
        preferred_candidate = process.extractOne(term, self.candidates, scorer=self.matching_method)
        score = preferred_candidate[1]

        if score > self.matching_threshold:
            return self.convert_term(preferred_candidate[0]), score
        else:
            return '', score


class ManbyoICDNormalizer(ManbyoNormalizer):

    def convert_term(self, term):
        ret = self.database.getTermICDCode(term)
        return 'NO_ICD_' + term if pd.isna(ret) else ret


class ManbyoMedDRANormalizer(ManbyoNormalizer):

    def convert_term(self, term):
        ret = self.database.getTermMedDRA(term)
        return 'NO_MEDDRA_MATCH' if pd.isna(ret) else ret
