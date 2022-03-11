import umls_api


def link_list_of_terms(list_entities):
    cuis = list()
    for entity in list_entities:
        results = umls_api.API(api_key='').term_search(entity)
        try:
            i = i + 1
            cui = results['result']['results'][0]['ui']
            print(cui)
        except Exception:
            cui = 0
        cuis.append(cui)
    return cuis


def link_single_term(entity):
    results = umls_api.API(api_key='').term_search(entity)
    return results['result']['results'][0]['ui']
