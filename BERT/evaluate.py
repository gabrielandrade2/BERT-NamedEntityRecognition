from seqeval.metrics import accuracy_score, precision_score, f1_score, classification_report
from seqeval.scheme import IOB2

from BERT.Model import NERModel
from util.list_utils import list_size


def evaluate(model: NERModel, test_sentences, test_labels):
    # Convert to BERT data model
    # test_x, test_y = bert_utils.dataset_to_bert_input(test_sentences, test_labels, model.tokenizer, model.vocabulary)

    # Normalize to same tokenization as BERT
    test_sentences, test_labels = model.normalize_tagged_dataset(test_sentences, test_labels)

    # Predict outputs
    data_x = model.prepare_sentences(test_sentences)
    predicted_labels = model.predict(data_x)
    data_x = model.convert_ids_to_tokens(data_x)

    # Evaluate model
    assert list_size(test_sentences) == list_size(data_x) == list_size(test_labels) == list_size(predicted_labels)
    print('Accuracy: ' + str(accuracy_score(test_labels, predicted_labels)))
    print('Precision: ' + str(precision_score(test_labels, predicted_labels)))
    print('F1 score: ' + str(f1_score(test_labels, predicted_labels)))
    print(classification_report(test_labels, predicted_labels, mode='strict', scheme=IOB2))
