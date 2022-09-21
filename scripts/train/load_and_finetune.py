from sklearn.model_selection import train_test_split

from BERT import bert_utils
from BERT.Model import NERModel
from util.xml_parser import convert_xml_file_to_iob_list

if __name__ == '__main__':
    # Load BERT model
    model = NERModel.load_transformers_model('cl-tohoku/bert-base-japanese-char-v2', '../../out/out_IM_v6')
    tokenizer = model.tokenizer
    label_vocab = model.vocabulary

    sentences, tags = convert_xml_file_to_iob_list('../../data/drugHistoryCheck.xml', ['d'],
                                                   should_split_sentences=True)

    # Convert 'd' tags to 'C'
    tags = [[t.replace('d', 'C') for t in l] for l in tags]

    ##### Split in train/validation #####
    train_x, validation_x, train_y, validation_y = train_test_split(sentences, tags, test_size=0.2)

    # Convert to BERT data model
    train_x, train_y = bert_utils.dataset_to_bert_input(train_x, train_y, tokenizer, label_vocab)
    validation_x, validation_y = bert_utils.dataset_to_bert_input(validation_x, validation_y, tokenizer, label_vocab)
    model.train(train_x, train_y, val=[validation_x, validation_y], outputdir='../out/finetunedmodel')
