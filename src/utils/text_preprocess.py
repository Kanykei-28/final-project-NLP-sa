import re

_HTML_RE = re.compile(r"<.*?>")
_NEG_CONTRACTIONS = [
    (r"can't", "can not"),
    (r"won't", "will not"),
    (r"n't", " not"),]

def basic_clean(text: str) -> str:
    text = text.lower()
    text = _HTML_RE.sub(" ", text)
    for pat, rep in _NEG_CONTRACTIONS:
        text = re.sub(pat, rep, text)
    return text