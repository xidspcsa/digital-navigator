import concurrent.futures
from typing import List

from dt_nav.api import settings
from dt_nav.models.document_keyword import DocumentKeywordStatus
from dt_nav.nlp.preprocess import CunningTokenizer
from dt_nav.processes.ner.jsonl_common import JsonlDatumStatus
from dt_nav.utils import st_preds_to_jsonl_datum
from tqdm import tqdm

from .model_common import get_trained_ner

__all__ = ["extract_entities", "extract_entities_many"]


def extract_entities(text: str) -> JsonlDatumStatus:
    """Extract named entities from the text

    Parameters
    ----------
    text : str
        A text to extract entities from

    Returns
    -------
    JsonlDatum
    """
    tokenizer = CunningTokenizer()
    sent_data = tokenizer.extract_sentences(text)
    tokenized_sent_data = [tokenizer.tokenize(s) for s in sent_data]
    tokenized_sentences = [t[0] for t in tokenized_sent_data]

    with get_trained_ner() as model:
        preds, _ = model.predict(tokenized_sentences, split_on_space=False)

    datum = st_preds_to_jsonl_datum(text, tokenized_sent_data, preds)
    datum["status"] = {}
    for e in datum["entities"]:
        datum["status"][text[e[0] : e[1]]] = DocumentKeywordStatus.EXTRACTED
    return datum


_tokenizer = None


def _init_pool():
    global _tokenizer
    _tokenizer = CunningTokenizer()


def _tokenize_text(text):
    global _tokenizer
    sent_data = _tokenizer.extract_sentences(text)
    tokenized_sent_data = [_tokenizer.tokenize(s) for s in sent_data]
    return tokenized_sent_data


def extract_entities_many(texts: List[str]):
    tokenized_sentences_all = []
    with concurrent.futures.ProcessPoolExecutor(
        initializer=_init_pool, max_workers=settings.device.max_workers
    ) as executor:
        with tqdm(total=len(texts), desc="Tokenizing") as bar:
            tokenized_sentences_all = []
            for s in executor.map(_tokenize_text, texts):
                tokenized_sentences_all.append(s)
                bar.update(1)

    res = []
    with get_trained_ner() as model:
        for text, tokenized_sent_data in tqdm(
            zip(texts, tokenized_sentences_all),
            total=len(texts),
            desc="Extracting entities",
           ):
            if len(tokenized_sent_data) == 0:
                res.append({"text": text, "entities": [], "status": {}})
                continue
            preds, _ = model.predict(
                [t[0] for t in tokenized_sent_data], split_on_space=False
            )
            datum = st_preds_to_jsonl_datum(text, tokenized_sent_data, preds)
            datum["status"] = {}
            for e in datum["entities"]:
                datum["status"][text[e[0] : e[1]]] = DocumentKeywordStatus.EXTRACTED

            res.append(datum)
    return res
