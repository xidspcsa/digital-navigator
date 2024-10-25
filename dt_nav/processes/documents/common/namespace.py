from flask_restx import Namespace, Resource, fields

from .document_status import get_document_status
from .documents_common import DOCUMENT_NEEDLE_PARAMS
from .find_dupes import get_dupes_data

__all__ = ["api"]

api = Namespace("documents/common", description="Работа с документами")


document_status_model = api.model(
    "DocumentStatus",
    {
        "id": fields.Integer(description="ID документа в системе"),
        "object_type": fields.String(description="Тип документа"),
        "system_id": fields.String(description="ID документа во внешней системе"),
        "root_id": fields.Integer(
            description="ID оригинального документа. Если не null, то данный документ считается дубликатом оригинального"
        ),
        "is_active": fields.Boolean(description="Документ активен"),
        "created_at": fields.DateTime(description="Дата создания документа"),
        "updated_at": fields.DateTime(description="Дата обновления документа"),
        "ner_extracted": fields.Boolean(
            description="Извлечены ли именованные сущности из последней версии документа"
        ),
    },
)

document_dupe_model = api.model(
    "Document",
    {
        "id": fields.Integer(description="ID документа в системе"),
        "object_type": fields.String(description="Тип документа"),
        "system_id": fields.String(description="ID документа во внешней системе"),
    },
)

dupes_model = api.model(
    "Dupes",
    {
        "root_document": fields.Nested(
            document_dupe_model, description="Оригинальный документ"
        ),
        "dupes": fields.List(
            fields.Nested(document_dupe_model), description="Список дубликатов"
        ),
    },
)


@api.route("/status/<string:needle>")
@api.param("needle", **DOCUMENT_NEEDLE_PARAMS)
class DocumentSync(Resource):
    @api.doc(description="Получить статус синхронизации документа")
    @api.marshal_with(document_status_model)
    def get(self, needle):
        return get_document_status(needle)


@api.route("/dupes/<string:needle>")
@api.param("needle", **DOCUMENT_NEEDLE_PARAMS)
class DocumentDupes(Resource):
    @api.doc(description="Получить дубликаты документа")
    @api.marshal_with(dupes_model)
    def get(self, needle):
        root_document, dupes_data = get_dupes_data(needle)
        return {
            "root_document": root_document.to_dict(
                include=["id", "system_id", "object_type"]
            ),
            "dupes": [
                d.to_dict(include=["id", "system_id", "object_type"])
                for d in dupes_data
            ],
        }
