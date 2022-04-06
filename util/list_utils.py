def flatten_list(l: list):
    return [item for sublist in l for item in sublist]


def list_size(l: list):
    return sum([len(t) for t in l])
