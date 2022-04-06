import unicodedata

from tqdm import tqdm

from BERT.Model import NERModel
from BERT.predict import predict_from_sentences_list
from util import iob_util, text_utils, list_utils
from util.Dataset import YakurekiTxtDataset
from util.evaluation import score
from util.list_utils import flatten_list

if __name__ == '__main__':
    dataset = YakurekiTxtDataset("/Users/gabriel-he/Documents/datasets/薬歴_タグ付け済_中江")
    model = NERModel.load_transformers_model('cl-tohoku/bert-base-japanese-char-v2', '../out/out_IM_v6')

    # dataset = ['患者像(70代男性、無職、尿酸値7.8mg/dl 既往歴 高血圧 軽度腎障害Ccr<30)\
    # 併用薬(<m-key>エナラプリル</m-key> <m-key>スピロノラクトン</m-key>)アドヒアランス良好\
    # S「<m-key>アロプリノール</m-key>を使ってたけど<d>最近腎臓の機能が弱まってる</d>と言われて、薬が変わることになった。\
    # 肝臓?それは問題ない。\
    # 」\
    # ', ]

    gold_tags = list()
    predicted_tags = list()
    count = 0
    for tagged_text in tqdm(dataset):
        tagged_text = unicodedata.normalize('NFKC', tagged_text)
        tagged_sentences = tagged_text.split('\n')
        tagged_sentences = [t for t in tagged_sentences if not t.startswith('既往歴')]

        gold = iob_util.convert_xml_text_list_to_iob_list(tagged_sentences, 'd', ignore_mismatch_tags=False)
        gold = model.normalize_tagged_dataset(gold[0], gold[1])

        sentences = text_utils.remove_tags(tagged_text).split('\n')
        sentences = [t for t in sentences if not t.startswith('既往歴')]
        predicted = predict_from_sentences_list(model, sentences)

        # Convert d to C for comparison
        predicted_temp = [[tag.replace('C', 'd') for tag in tag_list] for tag_list in predicted[1]]
        gold_temp = gold[1]

        if list_utils.list_size(gold_temp) == list_utils.list_size(predicted_temp):
            predicted_tags.extend(predicted_temp)
            gold_tags.extend(gold_temp)

            print(flatten_list(gold[0]))
            gold_temp = flatten_list(gold_temp)
            print(gold_temp)
            predicted_temp = flatten_list(predicted_temp)
            print(predicted_temp)
            print(score(gold_temp, predicted_temp))
        else:
            count += 1

    print('Ignored {}/{} texts'.format(count, len(dataset)))
    iob_util.evaluate_performance(gold_tags, predicted_tags)
