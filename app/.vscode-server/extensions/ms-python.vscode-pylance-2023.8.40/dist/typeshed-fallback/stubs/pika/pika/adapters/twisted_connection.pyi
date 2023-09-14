# twisted is optional and self-contained in this module.
# We don't want to force it as a dependency but that means we also can't test it with type-checkers given the current setup.

from _typeshed import Incomplete
from typing import NamedTuple, TypeVar

import pika.connection
from pika.adapters.utils import nbio_interface
from twisted.internet.base import DelayedCall  # type: ignore[import]  # pyright: ignore[reportMissingImports]
from twisted.internet.defer import Deferred, DeferredQueue  # type: ignore[import]  # pyright: ignore[reportMissingImports]
from twisted.internet.interfaces import ITransport  # type: ignore[import]  # pyright: ignore[reportMissingImports]
from twisted.internet.protocol import Protocol  # type: ignore[import]  # pyright: ignore[reportMissingImports]
from twisted.python.failure import Failure  # type: ignore[import]  # pyright: ignore[reportMissingImports]

_T = TypeVar("_T")

LOGGER: Incomplete

class ClosableDeferredQueue(DeferredQueue[_T]):  # pyright: ignore[reportUntypedBaseClass]
    closed: Failure | BaseException | None
    def __init__(self, size: Incomplete | None = ..., backlog: Incomplete | None = ...) -> None: ...
    # Returns a Deferred with an error if fails. None if success
    def put(self, obj: _T) -> Deferred[Failure | BaseException] | None: ...  # type: ignore[override]  # pyright: ignore[reportInvalidTypeVarUse]
    def get(self) -> Deferred[Failure | BaseException | _T]: ...
    pending: Incomplete
    def close(self, reason: BaseException | None) -> None: ...

class ReceivedMessage(NamedTuple):
    channel: Incomplete
    method: Incomplete
    properties: Incomplete
    body: Incomplete

class TwistedChannel:
    on_closed: Deferred[Incomplete | Failure | BaseException | None]
    def __init__(self, channel) -> None: ...
    @property
    def channel_number(self): ...
    @property
    def connection(self): ...
    @property
    def is_closed(self): ...
    @property
    def is_closing(self): ...
    @property
    def is_open(self): ...
    @property
    def flow_active(self): ...
    @property
    def consumer_tags(self): ...
    def callback_deferred(self, deferred, replies) -> None: ...
    def add_on_return_callback(self, callback): ...
    def basic_ack(self, delivery_tag: int = ..., multiple: bool = ...): ...
    def basic_cancel(self, consumer_tag: str = ...) -> Deferred[Incomplete | Failure | BaseException | None]: ...
    def basic_consume(
        self,
        queue,
        auto_ack: bool = ...,
        exclusive: bool = ...,
        consumer_tag: Incomplete | None = ...,
        arguments: Incomplete | None = ...,
    ) -> Deferred[Incomplete | Failure | BaseException]: ...
    def basic_get(self, queue, auto_ack: bool = ...) -> Deferred[Incomplete | Failure | BaseException | None]: ...
    def basic_nack(self, delivery_tag: Incomplete | None = ..., multiple: bool = ..., requeue: bool = ...): ...
    def basic_publish(
        self, exchange, routing_key, body, properties: Incomplete | None = ..., mandatory: bool = ...
    ) -> Deferred[Incomplete | Failure | BaseException]: ...
    def basic_qos(
        self, prefetch_size: int = ..., prefetch_count: int = ..., global_qos: bool = ...
    ) -> Deferred[Incomplete | Failure | BaseException | None]: ...
    def basic_reject(self, delivery_tag, requeue: bool = ...): ...
    def basic_recover(self, requeue: bool = ...) -> Deferred[Incomplete | Failure | BaseException | None]: ...
    def close(self, reply_code: int = ..., reply_text: str = ...): ...
    def confirm_delivery(self) -> Deferred[Incomplete | None]: ...
    def exchange_bind(
        self, destination, source, routing_key: str = ..., arguments: Incomplete | None = ...
    ) -> Deferred[Incomplete | Failure | BaseException | None]: ...
    def exchange_declare(
        self,
        exchange,
        exchange_type=...,
        passive: bool = ...,
        durable: bool = ...,
        auto_delete: bool = ...,
        internal: bool = ...,
        arguments: Incomplete | None = ...,
    ) -> Deferred[Incomplete | Failure | BaseException | None]: ...
    def exchange_delete(
        self, exchange: Incomplete | None = ..., if_unused: bool = ...
    ) -> Deferred[Incomplete | Failure | BaseException | None]: ...
    def exchange_unbind(
        self,
        destination: Incomplete | None = ...,
        source: Incomplete | None = ...,
        routing_key: str = ...,
        arguments: Incomplete | None = ...,
    ) -> Deferred[Incomplete | Failure | BaseException | None]: ...
    def flow(self, active) -> Deferred[Incomplete | Failure | BaseException | None]: ...
    def open(self): ...
    def queue_bind(
        self, queue, exchange, routing_key: Incomplete | None = ..., arguments: Incomplete | None = ...
    ) -> Deferred[Incomplete | Failure | BaseException | None]: ...
    def queue_declare(
        self,
        queue,
        passive: bool = ...,
        durable: bool = ...,
        exclusive: bool = ...,
        auto_delete: bool = ...,
        arguments: Incomplete | None = ...,
    ) -> Deferred[Incomplete | Failure | BaseException | None]: ...
    def queue_delete(
        self, queue, if_unused: bool = ..., if_empty: bool = ...
    ) -> Deferred[Incomplete | Failure | BaseException | None]: ...
    def queue_purge(self, queue) -> Deferred[Incomplete | Failure | BaseException | None]: ...
    def queue_unbind(
        self, queue, exchange: Incomplete | None = ..., routing_key: Incomplete | None = ..., arguments: Incomplete | None = ...
    ) -> Deferred[Incomplete | Failure | BaseException | None]: ...
    def tx_commit(self) -> Deferred[Incomplete | Failure | BaseException | None]: ...
    def tx_rollback(self) -> Deferred[Incomplete | Failure | BaseException | None]: ...
    def tx_select(self) -> Deferred[Incomplete | Failure | BaseException | None]: ...

class _TwistedConnectionAdapter(pika.connection.Connection):
    def __init__(self, parameters, on_open_callback, on_open_error_callback, on_close_callback, custom_reactor) -> None: ...
    def connection_made(self, transport: ITransport) -> None: ...
    def connection_lost(self, error: Exception) -> None: ...
    def data_received(self, data) -> None: ...

class TwistedProtocolConnection(Protocol):  # pyright: ignore[reportUntypedBaseClass]
    ready: Deferred[None] | None
    closed: Deferred[None] | Failure | BaseException | None
    def __init__(self, parameters: Incomplete | None = ..., custom_reactor: Incomplete | None = ...) -> None: ...
    def channel(self, channel_number: Incomplete | None = ...): ...
    @property
    def is_open(self): ...
    @property
    def is_closed(self): ...
    def close(self, reply_code: int = ..., reply_text: str = ...) -> Deferred[None] | Failure | BaseException | None: ...
    def dataReceived(self, data) -> None: ...
    def connectionLost(self, reason: Failure | BaseException = ...) -> None: ...
    def makeConnection(self, transport: ITransport) -> None: ...
    def connectionReady(self): ...

class _TimerHandle(nbio_interface.AbstractTimerReference):
    def __init__(self, handle: DelayedCall) -> None: ...
    def cancel(self) -> None: ...
