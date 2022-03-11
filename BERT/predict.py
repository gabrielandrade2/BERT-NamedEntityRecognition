def predict_from_sentences_list(model, sentences, split_sentences=False):
    sentences_embeddings = model.prepare_sentences(sentences, split_sentences)
    labels = model.predict(sentences_embeddings)
    sentences = model.convert_ids_to_tokens(sentences_embeddings)
    return sentences, labels
