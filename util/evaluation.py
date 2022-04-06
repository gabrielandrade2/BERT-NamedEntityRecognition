from pprint import pprint

from util import iob_util


def score(gold, predicted):
    # gold = list_utils.flatten_list(gold)
    # predicted = list_utils.flatten_list(predicted)
    gold = iob_util.convert_iob_taglist_to_dict(gold)
    gold = sorted(gold, key=lambda x: x['span'][0])
    predicted = iob_util.convert_iob_taglist_to_dict(predicted)
    predicted = sorted(predicted, key=lambda x: x['span'][0])

    score = 0
    matched = set()
    results = dict.fromkeys(
        ["exact_match", "exceeding_match", "exceeding_match_overlap", "partial_match", "partial_match_overlap",
         "sub_match", "missing_match", "incorrect_match"], 0)

    for match in predicted:
        found_gold_match = False
        pred_start, pred_end = match['span']
        for i in range(len(gold)):
            gold_start, gold_end = gold[i]['span']

            # Skip gold tags that came before
            if pred_start >= gold_end:
                continue

            # We can stop searching now
            elif pred_end <= gold_start:
                break

            # Exact match
            elif match['span'] == gold[i]['span']:
                score += 3
                results["exact_match"] += 1
                matched.add(gold[i]['span'])
                found_gold_match = True

            # Exceeding match, but not overlap next entity
            elif pred_start <= gold_start and pred_end >= gold_end:
                if i + 1 < len(gold):
                    if pred_end < gold[i + 1]['span'][0]:  # next gold start
                        score += 2
                        results["exceeding_match"] += 1
                        matched.add(gold[i]['span'])
                        found_gold_match = True
                    else:
                        score -= 1
                        results["exceeding_match_overlap"] += 1
                        matched.add(gold[i]['span'])
                        found_gold_match = True
                else:
                    score += 2
                    results["exceeding_match"] += 1
                    matched.add(gold[i]['span'])
                    found_gold_match = True

            # Partial match
            elif (pred_start >= gold_start and pred_end >= gold_end) or (
                    pred_start <= gold_start and pred_end <= gold_end):
                if i + 1 < len(gold):
                    if pred_end < gold[i + 1]['span'][0]:  # next gold start
                        score += 0
                        results["partial_match"] += 1
                        matched.add(gold[i]['span'])
                        found_gold_match = True
                    else:
                        score -= 1
                        results["partial_match_overlap"] += 1
                        matched.add(gold[i]['span'])
                        found_gold_match = True
                else:
                    score += 0
                    results["partial_match"] += 1
                    matched.add(gold[i]['span'])
                    found_gold_match = True

            # Sub-match
            elif pred_start >= gold_start and pred_end <= gold_end:
                score += 0
                results["sub_match"] += 1
                matched.add(gold[i]['span'])
                found_gold_match = True

            else:
                raise Exception("Insanity-check, should not be here!")

        if not found_gold_match:
            score -= 1
            results["incorrect_match"] += 1

    missing_match = len(gold) - len(matched)
    score -= 3 * missing_match
    results["missing_match"] = missing_match
    pprint(results)
    if len(gold):
        return score / (len(gold) * 3)
    return score


if __name__ == '__main__':
    O = 'O'
    B = 'B'
    I = 'I'

    gold = [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    print(gold)
    print(test)
    print(score(gold, test))
    print('\n')

    #      [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, B, I, I, I, I, O, O, O, B, I, I, I, I, I, O, O]
    print(gold)
    print(test)
    print(score(gold, test))
    print('\n')

    #      [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, B, I, O, B, I, O, O, O, O, B, I, I, I, I, O, O]
    print(gold)
    print(test)
    print(score(gold, test))
    print('\n')

    #      [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, B, I, O, O, O, O, O, B, I, O, O, O, O, O, O, O]
    print(gold)
    print(test)
    print(score(gold, test))
    print('\n')

    #      [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, B, I, O, B, I, O, B, I, O, B, I, O, B, I, O, O]
    print(gold)
    print(test)
    print(score(gold, test))
    print('\n')

    #      [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, B, I, I, I, I, I, I, I, I, I, I, I, I, I, O, O]
    print(gold)
    print(test)
    print(score(gold, test))
    print('\n')

    #      [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, B, I, I, I, I, I, I, O, O, B, I, I, I, I, I, I]
    print(gold)
    print(test)
    print(score(gold, test))
    print('\n')

    #      [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, O, O, B, I, I, O, O, O, O, O, B, I, I, O, O, O]
    print(gold)
    print(test)
    print(score(gold, test))
    print('\n')

    #      [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, B, I, I, I, I, I, I, I, I, I, I, O, B, I, I, I]
    print(gold)
    print(test)
    print(score(gold, test))
    print('\n')

    #      [O, O, B, I, I, I, O, O, O, O, B, I, I, I, I, O, O]
    test = [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, O]
    print(gold)
    print(test)
    print(score(gold, test))
    print('\n')
