# import nltk
# nltk.download('punkt')

# from nltk.tokenize import word_tokenize

POS_CUES_FOR_REL = [
    "is relevant",
    "are relevant",
    ": relevant",
    '"relevant"',
    "as relevant",
    "be relevant",
]

NEG_CUES_FOR_NONREL = [
    "is not relevant",
    "is nonrelevant",
    "are not relevant",
    "are nonrelevant",
    ": nonrelevant",
    ": not relevant",
    ": non",
    '"nonrelevant"',
    "as nonrelevant",
    "be nonrelevant",
]

def extract_label_from_llama_output(text):
    # text = " ".join(text.lower().split())
    assume_true, assume_false = False, False
    if any([x in text for x in POS_CUES_FOR_REL]):
        assume_true = True
    if any([x in text for x in NEG_CUES_FOR_NONREL]):
        assume_false = True
    if text.startswith("relevant."):
        assume_true = True
    if text.startswith("nonrelevant."):
        assume_false = True
    if assume_false:
        return False
    elif not assume_true and not assume_false:  # no cue extracted from text
        # for x in pos_clues:
        #     print(x, x in text)
        # for x in neg_clues:
        #     print(x, x in text)
        # print(assume_true, assume_false, text)
        # assert False
        return "error"
    elif assume_true:
        return True
    else:
        assert False