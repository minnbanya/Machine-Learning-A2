from typing import Any

from . import fields

RequestField = fields.RequestField

writer: Any

def choose_boundary(): ...
def iter_field_objects(fields): ...
def iter_fields(fields): ...
def encode_multipart_formdata(fields, boundary=...): ...
