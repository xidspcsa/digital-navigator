import logging

import dramatiq
import pandas as pd
from dt_nav.api import settings
from dt_nav.nlp.preprocess import CunningTokenizer
from dt_nav.tasks import broker
from dt_nav.utils import read_jsonl
from simpletransformers.ner import NERModel
from tqdm import tqdm

from .model_common import get_model_args

__all__ = ["train_ner"]


def _collect_jsonls_for_training():
    data_by_id = {}
    for f in settings.ner.train_files:
        data = read_jsonl(f)
        for datum in data:
            id_num = datum["id"]
            if datum["kind"] == "rpd":
                rpd_id = datum.get("meta", {}).get("id", None)
                if rpd_id is not None:
                    id_num = datum["meta"]["id"]
                else:
                    id_num = f"unk-{id_num}"
            id_ = f'{datum["kind"]}-{id_num}'
            datum["id"] = id_
            data_by_id[id_] = datum

    data = list(data_by_id.values())

    kinds = set(settings.ner.train_kinds)
    if len(kinds) > 0:
        filtered_data = []
        for datum in data:
            if datum["kind"] in kinds:
                filtered_data.append(datum)
        logging.info(f"Filtered {len(filtered_data)}/{len(data)} train documents")
        return filtered_data
    else:
        return data


def _jsonl_to_sentences_df(data):
    tokenizer = CunningTokenizer()
    res = []

    for datum in tqdm(data):
        doc_meta = {"id": datum["id"], "kind": datum["kind"]}

        sentences = tokenizer.extract_sentences(datum["text"], datum["entities"])
        sentences_with_entities = tokenizer.add_entities_to_sentences(
            sentences, datum["entities"]
        )
        for s in sentences_with_entities:
            tokens, tags, _ = tokenizer.tokenize(s, with_bio=True)
            res.append({**doc_meta, "tokens": tokens, "tags": tags})
    df = pd.DataFrame(res)
    return df


def _sentences_df_to_tokens_df(df_sent):
    res = []
    for i, td in enumerate(df_sent.itertuples()):
        for token, tag in zip(td.tokens, td.tags):
            res.append((td.id, i, token, tag))
    return pd.DataFrame(res, columns=["doc_id", "sentence_id", "words", "labels"])


def _eval_ner(model, df):
    kinds = df.kind.unique()
    for kind in kinds:
        df_eval = df[df.kind == kind]
        result, _, _ = model.eval_model(df_eval)
        logging.info(f"Eval on {kind}: \n {result}")


@dramatiq.actor(max_retries=0, broker=broker, time_limit=60 * 60 * 1000)
def train_ner():
    data = _collect_jsonls_for_training()
    df_sent = _jsonl_to_sentences_df(data)

    train_df = _sentences_df_to_tokens_df(df_sent)

    args = get_model_args()
    args.output_dir = settings.ner.state_dir

    model = NERModel(
        "bert",
        "DeepPavlov/rubert-base-cased",
        args=args,
        use_cuda=settings.ner.use_cuda,
    )
    logging.info(f"Starting training NER model")
    model.train_model(train_df)

    _eval_ner(model, train_df)
