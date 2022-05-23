import pandas as pd
from thefuzz import fuzz


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
        return self.df['出現形'].to_list()

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
