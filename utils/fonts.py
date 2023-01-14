from fontTools.ttLib import TTFont
from itertools import chain
from fontTools.unicode import Unicode


def load_chars(path):
    with open(path) as f:
        chars = f.read().split("\n")
        if chars[0].isdigit():
            chars = [chr(int(c)) for c in chars]
    return chars


def get_unicode_coverage_from_ttf(ttf_path):
    with TTFont(ttf_path, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1) as ttf:
        chars = chain.from_iterable([y + (Unicode[y[0]],) for y in x.cmap.items()] for x in ttf["cmap"].tables)
        chars_dec = [x[0] for x in chars]
        return chars_dec, [chr(x) for x in chars_dec]
