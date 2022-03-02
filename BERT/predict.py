from BERT import bert_utils

def predict_from_sentences_list(model, sentences):
    tokenizer = model.tokenizer
    sentences_embeddings = bert_utils.prepare_sentences(sentences, tokenizer)
    tags = model.predict(sentences_embeddings)
    labels = convert_prediction_to_labels(tags, model.vocabulary)
    sentences = [tokenizer.convert_ids_to_tokens(t)[1:] for t in sentences_embeddings]
    labels = remove_label_padding(sentences, labels)
    return sentences, labels

def convert_prediction_to_labels(prediction, vocabulary):
    id2label = {v: k for k, v in vocabulary.items()}
    return [[id2label[t] for t in tag] for tag in prediction]

def remove_label_padding(sentences, labels):
    new_labels = list()
    for sent, label in zip(sentences, labels):
        new_labels.append(label[:len(sent)])
    return new_labels
