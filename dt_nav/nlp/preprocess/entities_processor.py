import string

from pymystem3 import Mystem

from .cunning_tokenizer import CunningTokenizer

_CAST_TOKENS = {"с++": "c++", "с": "c", "C++": "c++", "C": "c"}

_STUPID_WORDS = set(
    [
        "знание",
        "владеть",
        "знать",
        "хороший",
        "отличный",
        "прекрасный",
        "и",
        "с",
        "умение",
        # "разработка",
        "понимание",
        "принцип",
        "опыт",
        "работа",
        "уметь",
        "навык",
        "готовность",
        "уверенный",
        "в",
        "роль",
        "знакомство",
        "желание",
        "базовый",
        "использование",
    ]
)

__all__ = ["EntitiesProcessor"]


class EntitiesProcessor:
    def __init__(self):
        self._tokenizer = CunningTokenizer()
        self._mystem = Mystem()

    def _preprocess_token_for_cnt(self, token):
        token = token.lower().strip()
        return token

    def preprocess_filter_empty(self, data):
        for datum in data:
            datum["entities"] = [e for e in datum["entities"] if e[0] < e[1]]

    def preprocess_cpp(self, data):
        for datum in data:
            datum["text"] = (
                datum["text"]
                .replace("С ++", "C++ ")
                .replace("C ++", "c++ ")
                .replace("С\n++", "С++ ")
                .replace("C\n++", "С++ ")
            )

    def preprocess_cast_tokens(self, data):
        for datum in data:
            text = datum["text"]
            for token_datum in datum["entities"]:
                start, end, _ = token_datum
                token = text[start:end]
                proc_token = self._preprocess_token_for_cnt(token)
                try:
                    casted = _CAST_TOKENS[proc_token]
                    if len(casted) < len(token):
                        casted = casted.ljust(len(token))
                    datum["text"] = text[:start] + casted + text[end:]
                    text = datum["text"]

                except KeyError:
                    pass

    def _convert_split(self, start, split, class_, tokens):
        res = []
        current_start = start
        for tok in split:
            if tok not in tokens and len(tok) > 0:
                res.append([current_start, current_start + len(tok), class_])
            current_start += len(tok)
        return res

    def _process_one_split(self, entities, text, func_, tokens):
        new_entities = []
        for token_datum in entities:
            start, end, class_ = token_datum
            token = text[start:end]

            split = func_(token)
            if split is not None and len(split) > 2:
                new_entities.extend(
                    self._convert_split(
                        start,
                        split,
                        class_,
                        tokens,
                    )
                )
            else:
                new_entities.append([start, end, class_])
        return new_entities

    def preprocess_split_tokens(self, data):
        for datum in data:
            text = datum["text"]
            for f, tokens in self._tokenizer.split_funcs:
                datum["entities"] = self._process_one_split(
                    datum["entities"],
                    text,
                    f,
                    tokens,
                )

    # XXX Потому что иногда он дописывает в конце бесполезный \n
    # И иногда прожевывает знаки препинания, например тут: Wi­Fi
    def _mystem_safe_analysis(self, token):
        analysis = self._mystem.analyze(token)
        len_ = 0
        res = []
        while len_ < len(token) and len(analysis) > 0:
            a = analysis.pop(0)
            res.append(a)
            len_ += len(a["text"])
        if len_ > len(token):
            res[-1]["text"] = res[-1]["text"][: (len(token) - len_)]
        if len(res) > 0 and res[-1]["text"][-1] == "\n":
            res[-1]["text"] = res[-1]["text"][:-1]
        return res

    def preprocess_prefix_tokens(self, data):
        for datum in data:
            text = datum["text"]
            for token_datum in datum["entities"]:
                start, end, _ = token_datum
                token = text[start:end]
                analysis = self._mystem_safe_analysis(token)

                actual_start = start
                if (
                    not self._tokenizer.is_english_token(token)
                    and len(token) > 5
                    and len(analysis) > 1
                    and not token[0:2].lower() in ["c+", "с+", "c#", "с#"]
                ):
                    for i, analyzed_token in enumerate(analysis):
                        try:
                            analysis_item = analyzed_token["analysis"][0]
                            if analysis_item["lex"] in _STUPID_WORDS:
                                actual_start += len(analyzed_token["text"])
                                continue
                        except (KeyError, IndexError):
                            pass
                        if (
                            analyzed_token["text"].isspace()
                            or analyzed_token["text"] in string.punctuation
                        ):
                            actual_start += len(analyzed_token["text"])
                            continue
                        break
                token_datum[0] = actual_start

                actual_end = end
                while actual_end > actual_start:
                    try:
                        last_char = text[actual_end - 1]
                    except IndexError:
                        break
                    if (
                        last_char in string.punctuation
                        and not last_char in [")", "+", "#"]
                        or last_char in string.whitespace
                    ):
                        actual_end -= 1
                    else:
                        break
                token_datum[1] = actual_end

    def process_tokens_override_classes(self, data):
        with open("./data/programming_languages.txt") as f:
            prog_lang_names = set([l.strip() for l in f.readlines()])

        for datum in data:
            text = datum["text"]
            for token_datum in datum["entities"]:
                start, end, _ = token_datum
                token = text[start:end]
                token = self._preprocess_token_for_cnt(token)
                if token in prog_lang_names:
                    token_datum[2] = "ProgLanguage"

    def process_tokens_freq(self, data):
        all_tokens = {}

        for datum in data:
            text = datum["text"]
            for start, end, class_ in datum["entities"]:
                token = text[start:end]
                token = self._preprocess_token_for_cnt(token)

                try:
                    token_data = all_tokens[token]
                except KeyError:
                    all_tokens[token] = {}
                    token_data = all_tokens[token]
                try:
                    token_data[class_] += 1
                except KeyError:
                    token_data[class_] = 1

        token_top_class = {}
        for token, token_data in all_tokens.items():
            top_class = max(token_data, key=token_data.get)
            token_top_class[token] = top_class

        for datum in data:
            text = datum["text"]
            for token_datum in datum["entities"]:
                start, end, _ = token_datum
                token = text[start:end]
                token = self._preprocess_token_for_cnt(token)
                token_datum[2] = token_top_class[token]

    def preprocess_punctuation(self, data):
        for datum in data:
            words = self._tokenizer.fix_punctuation_with_data(datum["text"])
            dumb_mapping = {}
            old_pos, new_pos = 0, 0
            for word, is_added in words:
                for _ in range(0, len(word)):
                    if not is_added:
                        dumb_mapping[old_pos] = new_pos
                    new_pos += 1
                    if not is_added:
                        old_pos += 1

            new_entities = []
            for start, end, t in datum["entities"]:
                new_entities.append((dumb_mapping[start], dumb_mapping[end], t))
            datum["text"] = "".join([w[0] for w in words])
            datum["entities"] = new_entities

    def preprocess_sentences(self, data, add_empty=True):
        for datum in data:
            sentences = self._tokenizer.extract_sentences(
                datum["text"], datum["entities"]
            )
            sentences_with_entities = self._tokenizer.add_entities_to_sentences(
                sentences, datum["entities"], add_empty=add_empty
            )
            new_text, new_entities = "", []
            for offset, sentence, sent_entities in sentences_with_entities:
                stripped_sentence = sentence.rstrip()
                if len(stripped_sentence) > 0:
                    if not stripped_sentence[-1] in string.punctuation:
                        stripped_sentence += "."
                    stripped_sentence += " "

                delta = offset - len(new_text)

                for entity in sent_entities:
                    start, end, ent_text = entity
                    new_entities.append((start - delta, end - delta, ent_text))

                new_text += stripped_sentence

            datum["text"] = new_text
            datum["entities"] = new_entities

    def process(self, data):
        self.preprocess_cpp(data)
        self.preprocess_split_tokens(data)
        self.preprocess_prefix_tokens(data)
        self.preprocess_cast_tokens(data)
        self.process_tokens_override_classes(data)
        self.process_tokens_freq(data)
        self.preprocess_filter_empty(data)
        self.preprocess_punctuation(data)
        self.preprocess_sentences(data)
        return data
