import re
import string
from typing import List, Tuple, Union

import numpy as np
import spacy

__all__ = ["CunningTokenizer"]

SentenceForTokenization = Union[
    str,
    Tuple[str],
    Tuple[int, str],
    Tuple[int, str, List[Tuple[int, int, str]]],
]


class CunningTokenizer:
    def __init__(self) -> None:
        self._nlp = spacy.load("ru_core_news_sm")

    def is_english_token(self, token):
        return re.search("[а-яА-Я]", token) is None

    def _maybe_split_by_slash(self, token):
        slash_split = re.split("([\\/])", token)
        if len(slash_split) < 2:
            return
        has_numbers = any([s.isnumeric() for s in slash_split])
        if has_numbers:
            return

        english_count = 0
        split_tokens = 0
        for s in slash_split:
            if s in ["\\", "/ ", " ", ""]:
                split_tokens += 1
            elif self.is_english_token(s):
                english_count += 1
        if english_count < (len(slash_split) - split_tokens) / 2:
            return
        return slash_split

    def _maybe_split_by_comma(self, token):
        comma_split = re.split("([,])", token)
        if len(comma_split) < 2:
            return
        commas = token.count(",")
        if commas > 6:
            return comma_split
        english_count = 0
        split_tokens = 0
        words = 0
        for s in comma_split:
            if s in [*string.whitespace, ",", ", "]:
                split_tokens += 1
            elif self.is_english_token(s):
                english_count += 1
            words += s.count(" ") + 1
        if english_count >= (len(comma_split) - split_tokens) / 2.5:
            return comma_split
        if words <= (len(comma_split) - split_tokens) * 2.5:
            return comma_split
        return

    def _maybe_split_by_semi(self, token):
        return re.split("([;\n])", token)

    @property
    def split_funcs(self):
        return [
            (self._maybe_split_by_semi, [";", "\n", *string.whitespace]),
            (self._maybe_split_by_comma, [",", *string.whitespace]),
            (self._maybe_split_by_slash, ["/", "\\", *string.whitespace]),
        ]

    def _extract_enums_by_punct(self, offset, s, sent_data, entities, i):
        res_s = ""
        lines = re.split("(\n)", s)

        new_sent_data = []

        while len(lines) > 0 and (lines[-1].isspace() or len(lines[-1]) == 0):
            lines = lines[:-1]

        line_offset = 0
        for j, line in enumerate(lines):
            is_last_line_with_entities = False
            is_line_with_colon = False
            if j == len(lines) - 1 and i == len(sent_data) - 1 and len(entities) > 0:
                last_entity_start, _, _ = entities[-1]
                if last_entity_start > line_offset + len(res_s):
                    is_last_line_with_entities = True

            is_line_with_colon = re.match("^[А-Я].*:\\s*$", line)

            if is_last_line_with_entities or is_line_with_colon:
                if not res_s.isspace():
                    new_sent_data.append((offset + line_offset, res_s))
                line_offset += len(res_s)
                res_s = ""

            res_s += line
        if not res_s.isspace():
            new_sent_data.append((offset + line_offset, res_s))
        return new_sent_data

    def _extract_enums_by_line_length(self, sent_data, line_threshold=55):
        new_sent_data = []
        for offset, s in sent_data:
            lines = re.split("(\n)", s)

            line_lengths = []

            for line in lines:
                if len(line) > 0 and not line.isspace():
                    line_lengths.append(len(line))

            if len(line_lengths) and np.mean(line_lengths) > line_threshold:
                line_offset = 0
                for line in lines:
                    if len(line) > 0 and not line.isspace():
                        new_sent_data.append((offset + line_offset, line))
                    line_offset += len(line)
            else:
                new_sent_data.append((offset, s))
        return new_sent_data

    def extract_sentences(self, text, entities=None) -> List[Tuple[int, str]]:
        if entities is None:
            entities = []
        sentences_by_dot = re.split("(\\.[ \n]|\n{3,}|\\!\\?)", text)
        offset = 0
        sent_data = []
        for s in sentences_by_dot:
            if not re.match("^[\\s\\.]*$", s):
                sent_data.append((offset, s))
            offset += len(s)

        new_sent_data = []
        for i, (offset, s) in enumerate(sent_data):
            local_sent_data = self._extract_enums_by_punct(
                offset, s, sent_data, entities, i
            )
            local_sent_data = self._extract_enums_by_line_length(local_sent_data)
            new_sent_data.extend(local_sent_data)

        new_sent_data = [s for s in new_sent_data if len(s[1]) > 0]

        return new_sent_data

    def add_entities_to_sentences(self, sentences, entities, add_empty=False):
        for i in range(len(sentences)):
            s = sentences[i]
            if len(s) == 3:
                sentences[i] = (s[0], s[1])

        if len(sentences) == 0 or len(entities) == 0:
            if add_empty:
                return [(o, s, []) for o, s in sentences]
            return []

        res, current_tokens = [], []
        sent_i = -1
        current_sent, current_offset = None, None
        current_sent_end = -1
        for start, end, class_ in entities:
            while start >= current_sent_end:
                if len(current_tokens) > 0 or add_empty and current_sent is not None:
                    res.append((current_offset, current_sent, current_tokens))
                    current_tokens = []
                sent_i += 1
                current_offset, current_sent = sentences[sent_i]
                current_sent_end = current_offset + len(current_sent)
            current_tokens.append((start, end, class_))

        while len(current_tokens) > 0 or add_empty and sent_i <= len(sentences):
            res.append((current_offset, current_sent, current_tokens))
            current_tokens = []
            sent_i += 1
            if sent_i < len(sentences):
                current_offset, current_sent = sentences[sent_i]
            else:
                break

        return res

    def fix_sentences(self, text):
        sent_data = self.extract_sentences(text, [])
        return ". ".join([s[1] for s in sent_data])

    def fix_punctuation_with_data(self, text):
        words = [(w, False) for w in re.split("(\\s+)", text)]
        res = []
        for f, tokens in self.split_funcs:
            for w, is_added in words:
                if w.isspace():
                    res.append((w, is_added))
                    continue

                split = f(w)
                if split is not None:
                    split = [s for s in split if len(s) > 0]
                if split is None or len(split) == 1:
                    res.append((w, is_added))
                else:
                    for i, s in enumerate(split):
                        if s in ["/", "\\"]:
                            res.append((" ", True))
                        res.append((s, is_added))
                        if s in tokens and i != len(split) - 1:
                            res.append((" ", True))
            words = res
            res = []
        return words

    def fix_punctuation(self, text):
        return "".join([w[0] for w in self.fix_punctuation_with_data(text)])

    def _tokenize_parameters(self, sentence: SentenceForTokenization):
        if isinstance(sentence, str):
            offset, text, entities = 0, sentence, []
        elif len(sentence) == 1:
            offset, text, entities = 0, sentence[0], []
        elif len(sentence) == 2:
            offset, text, entities = sentence[0], sentence[1], []
        else:
            offset, text, entities = sentence
        return offset, text, entities

    def tokenize(self, sentence: SentenceForTokenization, with_bio=False):
        offset, text, entities = self._tokenize_parameters(sentence)
        doc = self._nlp(text)

        ent_start, ent_end, ent_class = -1, -1, None
        ent_idx = -1

        tags, tokens, offsets = [], [], []
        for token in doc:
            token_offset = offset + token.idx
            if len(entities) > 0:
                while ent_end < token_offset:
                    ent_idx += 1
                    if ent_idx >= len(entities):
                        break
                    ent_start, ent_end, ent_class = entities[ent_idx]
                if ent_end > token_offset >= ent_start and not re.match(
                    "^\\s*\\n+\\s*$", token.text
                ):
                    if with_bio:
                        if token_offset == ent_start:
                            tags.append(f"B-{ent_class}")
                        else:
                            tags.append(f"I-{ent_class}")
                    else:
                        tags.append(ent_class)
                else:
                    tags.append("O")
            else:
                tags.append("O")
            tokens.append(token.text)
            offsets.append(token_offset)

        return tokens, tags, offsets

    def jsonl_datum_to_labels(self, text, entities):
        sent_data = self.extract_sentences(text)
        sent_data = self.add_entities_to_sentences(sent_data, entities, True)
        tokenized_sent_data = [self.tokenize(s) for s in sent_data]
        kw_labels = [t[1] for t in tokenized_sent_data]
        kw_tokens = [t[0] for t in tokenized_sent_data]
        return kw_labels, kw_tokens
