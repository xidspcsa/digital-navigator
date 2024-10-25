import json
from typing import List, Tuple, TypedDict

__all__ = [
    "JsonlDatum",
    "NEREntities",
    "read_jsonl",
    "write_jsonl",
    "st_preds_to_jsonl_datum",
    "jsonl_datum_to_annotated_text",
]

NEREntities = List[Tuple[int, int, str]]


class JsonlDatum(TypedDict):
    """A text with extracted entities

    "text" is the text, "entities" is the list of entities. The
    elements of the tuple are as follows: start, end, and entity
    class.
    """

    text: str
    entities: NEREntities


def read_jsonl(file_path) -> List[JsonlDatum]:
    with open(file_path, "r") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]


def write_jsonl(file_path: str, data: List[JsonlDatum]):
    with open(file_path, "w", encoding="utf-8") as f:
        for datum in data:
            f.write(json.dumps(datum, ensure_ascii=False))
            f.write("\n")


def st_preds_to_jsonl_datum(
    text, tokenized_sent_data, preds, with_bio=True
) -> JsonlDatum:
    entities = []
    for sentence, pred_sentence in zip(tokenized_sent_data, preds):
        tokens, _, offsets = sentence

        start, end, value = None, None, None
        for token, prediction, offset in zip(tokens, pred_sentence, offsets):
            key = list(prediction.keys())[0]
            curr_value = prediction[key]
            # if len(text) < offset:
            #     text = text.ljust(offset)
            # text += token

            if not with_bio:
                if curr_value != "O":
                    entities.append([offset, offset + len(token), value])
            else:
                if curr_value == "O":
                    if value is not None:
                        entities.append([start, end, value])
                        start, end, value = None, None, None
                else:
                    curr_value_pos, curr_value_tag = curr_value.split("-")
                    if (
                        curr_value_pos == "B"
                        or curr_value_pos == "I"
                        and curr_value_tag != value
                    ):
                        if value is not None:
                            entities.append([start, end, value])
                            start, end, value = None, None, None
                        start, end, value = offset, offset + len(token), curr_value_tag
                    elif curr_value_pos == "I" and curr_value_tag == value:
                        end = offset + len(token)
        if with_bio and value is not None:
            entities.append([start, end, value])

    return {"text": text, "entities": entities}


def jsonl_datum_to_annotated_text(datum: JsonlDatum):
    text = datum["text"]

    start_idx = 0
    res = []
    for start, end, class_ in datum["entities"]:
        res.append(text[start_idx:start])
        res.append((text[start:end], class_))
        start_idx = end
    res.append(text[start_idx:])
    return res
