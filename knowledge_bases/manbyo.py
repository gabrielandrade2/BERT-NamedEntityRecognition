import pandas as pd
from thefuzz import fuzz

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

    def searchTerm(self, search, num_candidates=1, return_scores=False):
        temp = [(i, fuzz.token_set_ratio(i, search)) for i in self.getTermList()]
        temp = sorted(temp, key=lambda x: x[1], reverse=True)[:num_candidates]
        if not return_scores:
            temp = list(map(lambda x: x[0], temp))
        return temp


def wrapper(method, pf, term):
    return (method(term, pf), pf)


class ManbyoNormalizer(EntityNormalizer):

    def __init__(self, database: ManbyoDict, matching_method=fuzz.token_set_ratio, threshold=0):
        self.database = database.getTermList()
        self.matching_method = matching_method
        self.threshold = threshold

    def normalize(self, term):
        temp = []
        for pf in self.database:
            score = self.matching_method(term, pf)
            temp.append((score, pf))
            if score == 100:
                break

        preferred_candidate = max(temp)

        score = preferred_candidate[0]

        if score > self.threshold:
            return preferred_candidate[1]
        else:
            return ''
