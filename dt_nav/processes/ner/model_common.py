import logging
from contextlib import contextmanager
from typing import ContextManager, List

from dt_nav.api import settings
from simpletransformers.ner import NERArgs, NERModel

__all__ = ["get_labels_list", "get_model_args", "get_trained_ner"]


def get_labels_list(with_bio=True) -> List[str]:
    """Get labels list from settings.

    Parameters
    ----------
    with_bio : bool
        If True, make labels for the BIO notation (default)

    Returns
    -------
    List[str]
    """

    res = []
    for l in settings.ner.labels:
        if l == "O" or not with_bio:
            res.append(l)
        else:
            res.append(f"B-{l}")
            res.append(f"I-{l}")
    return res


def get_model_args():
    args = NERArgs()
    args.classification_report = True
    args.labels_list = get_labels_list(with_bio=True)
    args.num_train_epochs = 5
    args.learning_rate = 1e-5
    args.overwrite_output_dir = True
    args.use_multiprocessing = False
    args.silent = True
    return args


_model = None


@contextmanager
def get_trained_ner() -> ContextManager[NERModel]:
    """Load model into memory.

    I suspect loading NERModel might have some overhead, so this
    attemtps to load it just once.

    Returns
    -------
    ContextManager[NERModel]


    Examples
    --------
    with get_trained_ner() as ner:
        ner.predict(["Hello world"])

    """
    global _model
    if _model is None:
        logging.info("Loading NER model")
        args = get_model_args()
        _model = NERModel(
            "bert", settings.ner.state_dir, args=args, use_cuda=settings.ner.use_cuda
        )
        try:
            yield _model
        finally:
            logging.info("Unloading NER model")
    else:
        yield _model
