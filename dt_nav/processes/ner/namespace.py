from dt_nav.utils import NerEntityElem
from dt_nav.utils.api_types import DictElem
from flask import request
from flask_restx import Namespace, Resource, fields, inputs

from .process_documents import get_saved_jsonl, save_jsonl_for_document

__all__ = ["api"]

api = Namespace("ner", description="Именованные сущности")


jsonl_model = api.model(
    "JsonlDatum",
    {
        "text": fields.String(description="Текст"),
        "entities": fields.List(
            NerEntityElem,
            description="Извлеченные сущности. Список кортежей вида (начало, конец, класс)",
            example=[[0, 10, "ProgLanguage"]],
        ),
        "status": DictElem(
            fields.String,
            description="Статусы ключевых слов",
            example={
                "keyword1": "extracted",
                "keyword2": "rejected",
                "keyword3": "added",
                "keyword4": "confirmed",
            },
        ),
    },
)


get_ner_document_parser = api.parser()
get_ner_document_parser.add_argument(
    "filter_rejected",
    type=inputs.boolean,
    default=True,
    required=False,
    location="args",
)


@api.route("/<string:object_type>/<string:document_id>")
@api.param("object_type", "Тип объекта", example="rpd")
@api.param("document_id", "ID документа")
class NerDocument(Resource):
    @api.marshal_with(jsonl_model)
    @api.doc(description="Получить именованные сущности для документа")
    @api.expect(get_ner_document_parser)
    def get(self, object_type: str, document_id: str):
        needle = f"{object_type}:{document_id}"
        args = get_ner_document_parser.parse_args()
        return get_saved_jsonl(needle, filter_rejected=args["filter_rejected"])

    @api.doc(description="Обновить именованные сущности для документа")
    @api.expect(jsonl_model)
    def put(self, object_type: str, document_id: str):
        needle = f"{object_type}:{document_id}"
        datum = request.json
        save_jsonl_for_document(needle, datum, is_user=True)
