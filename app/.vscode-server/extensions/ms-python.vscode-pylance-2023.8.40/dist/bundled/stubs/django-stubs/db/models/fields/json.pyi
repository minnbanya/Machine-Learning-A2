import json
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from django.db.models import lookups
from django.db.models.expressions import Combinable
from django.db.models.lookups import PostgresOperatorLookup, Transform
from typing_extensions import Literal

from . import Field, _ErrorMessagesToOverride, _ValidatorCallable
from .mixins import CheckFieldDefaultMixin

_A = TypeVar("_A", bound=Optional[Any])

class JSONField(CheckFieldDefaultMixin, Field[Union[_A, Combinable], _A]):
    default_error_messages: Any = ...
    encoder: Type[json.JSONEncoder] = ...
    decoder: Type[json.JSONEncoder] = ...
    def from_db_value(self, value: Any, expression: Any, connection: Any) -> Any: ...
    def get_transform(self, name: Any) -> Any: ...
    def value_to_string(self, obj: Any) -> Any: ...
    @overload
    def __new__(  # type: ignore [misc]
        cls,
        verbose_name: Optional[str] = ...,
        name: Optional[str] = ...,
        encoder: Type[json.JSONEncoder] = ...,
        decoder: Type[json.JSONDecoder] = ...,
        primary_key: bool = ...,
        max_length: Optional[int] = ...,
        unique: bool = ...,
        blank: bool = ...,
        null: Literal[False] = ...,
        db_index: bool = ...,
        default: Optional[Union[Any, Callable[[], Any]]] = ...,
        editable: bool = ...,
        auto_created: bool = ...,
        serialize: bool = ...,
        unique_for_date: Optional[str] = ...,
        unique_for_month: Optional[str] = ...,
        unique_for_year: Optional[str] = ...,
        choices: Iterable[
            Union[Tuple[Any, str], Tuple[str, Iterable[Tuple[Any, str]]]]
        ] = ...,
        help_text: str = ...,
        db_column: Optional[str] = ...,
        db_tablespace: Optional[str] = ...,
        validators: Iterable[_ValidatorCallable] = ...,
        error_messages: Optional[_ErrorMessagesToOverride] = ...,
    ) -> JSONField[_A]: ...
    @overload
    def __new__(
        cls,
        verbose_name: Optional[str] = ...,
        name: Optional[str] = ...,
        encoder: Type[json.JSONEncoder] = ...,
        decoder: Type[json.JSONDecoder] = ...,
        primary_key: bool = ...,
        max_length: Optional[int] = ...,
        unique: bool = ...,
        blank: bool = ...,
        null: Literal[True] = ...,
        db_index: bool = ...,
        default: Union[Any, Callable[[], Any]] = ...,
        editable: bool = ...,
        auto_created: bool = ...,
        serialize: bool = ...,
        unique_for_date: Optional[str] = ...,
        unique_for_month: Optional[str] = ...,
        unique_for_year: Optional[str] = ...,
        choices: Iterable[
            Union[Tuple[Any, str], Tuple[str, Iterable[Tuple[Any, str]]]]
        ] = ...,
        help_text: str = ...,
        db_column: Optional[str] = ...,
        db_tablespace: Optional[str] = ...,
        validators: Iterable[_ValidatorCallable] = ...,
        error_messages: Optional[_ErrorMessagesToOverride] = ...,
    ) -> JSONField[Optional[_A]]: ...

class DataContains(PostgresOperatorLookup):
    lookup_name: str = ...
    postgres_operator: str = ...
    def as_sql(self, compiler: Any, connection: Any) -> Any: ...

class ContainedBy(PostgresOperatorLookup):
    lookup_name: str = ...
    postgres_operator: str = ...
    def as_sql(self, compiler: Any, connection: Any) -> Any: ...

class HasKeyLookup(PostgresOperatorLookup):
    logical_operator: Any = ...
    def as_sql(
        self, compiler: Any, connection: Any, template: Optional[Any] = ...
    ) -> Any: ...
    def as_mysql(self, compiler: Any, connection: Any) -> Any: ...
    def as_oracle(self, compiler: Any, connection: Any) -> Any: ...
    lhs: Any = ...
    rhs: Any = ...
    def as_postgresql(self, compiler: Any, connection: Any) -> Any: ...
    def as_sqlite(self, compiler: Any, connection: Any) -> Any: ...

class HasKey(HasKeyLookup):
    lookup_name: str = ...
    postgres_operator: str = ...
    prepare_rhs: bool = ...

class HasKeys(HasKeyLookup):
    lookup_name: str = ...
    postgres_operator: str = ...
    logical_operator: str = ...
    def get_prep_lookup(self) -> Any: ...

class HasAnyKeys(HasKeys):
    lookup_name: str = ...
    postgres_operator: str = ...
    logical_operator: str = ...

class JSONExact(lookups.Exact):
    can_use_none_as_rhs: bool = ...
    def process_rhs(self, compiler: Any, connection: Any) -> Any: ...

class KeyTransform(Transform):
    postgres_operator: str = ...
    postgres_nested_operator: str = ...
    key_name: Any = ...
    def __init__(self, key_name: Any, *args: Any, **kwargs: Any) -> None: ...
    def preprocess_lhs(
        self, compiler: Any, connection: Any, lhs_only: bool = ...
    ) -> Any: ...
    def as_mysql(self, compiler: Any, connection: Any) -> Any: ...
    def as_oracle(self, compiler: Any, connection: Any) -> Any: ...
    def as_postgresql(self, compiler: Any, connection: Any) -> Any: ...

class KeyTextTransform(KeyTransform):
    postgres_operator: str = ...
    postgres_nested_operator: str = ...

class KeyTransformTextLookupMixin:
    def __init__(self, key_transform: Any, *args: Any, **kwargs: Any) -> None: ...

class CaseInsensitiveMixin:
    def process_rhs(self, compiler: Any, connection: Any) -> Any: ...

class KeyTransformIsNull(lookups.IsNull):
    def as_oracle(self, compiler: Any, connection: Any) -> Any: ...
    def as_sqlite(self, compiler: Any, connection: Any) -> Any: ...

class KeyTransformIn(lookups.In):
    def process_rhs(self, compiler: Any, connection: Any) -> Any: ...

class KeyTransformExact(JSONExact):
    def process_rhs(self, compiler: Any, connection: Any) -> Any: ...
    def as_oracle(self, compiler: Any, connection: Any) -> Any: ...

class KeyTransformIExact(
    CaseInsensitiveMixin, KeyTransformTextLookupMixin, lookups.IExact
): ...
class KeyTransformIContains(
    CaseInsensitiveMixin, KeyTransformTextLookupMixin, lookups.IContains
): ...
class KeyTransformStartsWith(KeyTransformTextLookupMixin, lookups.StartsWith): ...
class KeyTransformIStartsWith(
    CaseInsensitiveMixin, KeyTransformTextLookupMixin, lookups.IStartsWith
): ...
class KeyTransformEndsWith(KeyTransformTextLookupMixin, lookups.EndsWith): ...
class KeyTransformIEndsWith(
    CaseInsensitiveMixin, KeyTransformTextLookupMixin, lookups.IEndsWith
): ...
class KeyTransformRegex(KeyTransformTextLookupMixin, lookups.Regex): ...
class KeyTransformIRegex(
    CaseInsensitiveMixin, KeyTransformTextLookupMixin, lookups.IRegex
): ...

class KeyTransformNumericLookupMixin:
    def process_rhs(self, compiler: Any, connection: Any) -> Any: ...

class KeyTransformLt(KeyTransformNumericLookupMixin, lookups.LessThan[Any]): ...
class KeyTransformLte(KeyTransformNumericLookupMixin, lookups.LessThanOrEqual): ...
class KeyTransformGt(KeyTransformNumericLookupMixin, lookups.GreaterThan): ...
class KeyTransformGte(
    KeyTransformNumericLookupMixin, lookups.GreaterThanOrEqual[Any]
): ...

class KeyTransformFactory:
    key_name: Any = ...
    def __init__(self, key_name: Any) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

KT: KeyTextTransform = ...
