from typing import (
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from django.core.checks.messages import CheckMessage
from django.core.exceptions import (
    MultipleObjectsReturned as BaseMultipleObjectsReturned,
)
from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db.models.manager import BaseManager
from django.db.models.options import Options

_M = TypeVar("_M", bound=Any)
_Self = TypeVar("_Self", bound="Model")

class ModelStateFieldsCacheDescriptor: ...

class ModelState:
    db: Optional[str] = ...
    adding: bool = ...
    fields_cache: ModelStateFieldsCacheDescriptor = ...

class ModelBase(type):
    # FIXME: It would be better to use _Self instead of _M here,
    # but pyright says Type[_Self] cannot be assigned here... Maybe a bug in pyright?
    @property
    def objects(cls: Type[_M]) -> BaseManager[_M]: ...
    @property
    def _meta(cls: Type[_M]) -> Options[_M]: ...
    @property
    def _default_manager(cls: Type[_M]) -> BaseManager[_M]: ...

class Model(metaclass=ModelBase):
    DoesNotExist: ClassVar[type[ObjectDoesNotExist]]
    MultipleObjectsReturned: ClassVar[type[BaseMultipleObjectsReturned]]

    class Meta: ...
    pk: Any = ...
    _state: ModelState
    _meta: Options[Any]
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @classmethod
    def add_to_class(cls, name: str, value: Any) -> Any: ...
    @classmethod
    def from_db(
        cls: Type[_Self],
        db: Optional[str],
        field_names: Collection[str],
        values: Collection[Any],
    ) -> _Self: ...
    def delete(
        self, using: Any = ..., keep_parents: bool = ...
    ) -> Tuple[int, Dict[str, int]]: ...
    async def adelete(
        self, using: Any = ..., keep_parents: bool = ...
    ) -> Tuple[int, Dict[str, int]]: ...
    def full_clean(
        self, exclude: Optional[Collection[str]] = ..., validate_unique: bool = True, validate_constraints: bool = True
    ) -> None: ...
    def clean(self) -> None: ...
    def clean_fields(self, exclude: Optional[Collection[str]] = ...) -> None: ...
    def validate_unique(self, exclude: Optional[Collection[str]] = ...) -> None: ...
    def unique_error_message(
        self: _Self,
        model_class: Type[_Self],
        unique_check: Collection[Union[Callable[..., Any], str]],
    ) -> ValidationError: ...
    def save(
        self,
        force_insert: bool = ...,
        force_update: bool = ...,
        using: Optional[str] = ...,
        update_fields: Optional[Iterable[str]] = ...,
    ) -> None: ...
    async def asave(
        self,
        force_insert: bool = ...,
        force_update: bool = ...,
        using: Optional[str] = ...,
        update_fields: Optional[Iterable[str]] = ...,
    ) -> None: ...
    def save_base(
        self,
        raw: bool = ...,
        force_insert: bool = ...,
        force_update: bool = ...,
        using: Optional[str] = ...,
        update_fields: Optional[Iterable[str]] = ...,
    ) -> Any: ...
    def refresh_from_db(
        self, using: Optional[str] = ..., fields: Optional[List[str]] = ...
    ) -> None: ...
    async def arefresh_from_db(
        self, using: Optional[str] = ..., fields: Optional[List[str]] = ...
    ) -> None: ...
    def get_deferred_fields(self) -> Set[str]: ...
    @classmethod
    def check(cls, **kwargs: Any) -> List[CheckMessage]: ...
    def __getstate__(self) -> Dict[Any, Any]: ...
