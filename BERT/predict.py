def predict_from_sentences_list(model, sentences, split_sentences=False, display_progress=False):
    sentences_embeddings = model.prepare_sentences(sentences, split_sentences)
    labels = model.predict(sentences_embeddings, display_progress=display_progress)
    sentences = model.convert_ids_to_tokens(sentences_embeddings)
    labels = [[l if l != "[PAD]" else "O" for l in label] for label in labels]
    return sentences, labels
