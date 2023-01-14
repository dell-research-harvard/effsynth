

def to_string_list(x):
    return [str(x) for x in list(x)]


def safe_list_get(l, idx, default):
    try:
        return l[idx]
    except IndexError:
        return default