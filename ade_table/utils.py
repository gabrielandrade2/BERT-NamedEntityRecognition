from dnorm_j import DNorm
from util.bert import iob_util


def convert_labels_to_dict(sentences, labels):
    ne_dict = list()
    for sent, label in zip(sentences, labels):
        ne_dict.extend(iob_util.convert_iob_to_dict(sent, label))
    return ne_dict

def normalize_entities(named_entities):
    normalized_entities = list()
    normalization_model =   DNorm.from_pretrained()
    for entry in named_entities:
        entry['normalized_word'] = normalization_model.normalize(entry['word'])
        normalized_entities.append(entry)
    return normalized_entities

def consolidate_table_data(drug, output_dict, ne_dict):
    if drug in output_dict:
        drug_dict = output_dict[drug]
    else:
        drug_dict = {}

    for named_entity in ne_dict:
        word = named_entity['normalized_word']
        if word in drug_dict:
            count = drug_dict[word] + 1
        else:
            count = 1
        drug_dict[word] = count
    output_dict[drug] = drug_dict
    return output_dict

def table_post_process(table):
    # Order drugs by number of ADE events
    table['sum_col'] = table.sum(axis=1)
    table.sort_values('sum_col', axis=0, ascending=False, inplace=True)
    table.drop(columns=["sum_col"], inplace=True)
    table = table[:50]

    # Order ADE by numer of events
    table = table[table.sum(0).sort_values(ascending=False)[:50].index]

    return table
