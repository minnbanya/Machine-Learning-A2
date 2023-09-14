from _typeshed import Incomplete
from typing import Any

from braintree.error_result import ErrorResult as ErrorResult
from braintree.exceptions.not_found_error import NotFoundError as NotFoundError
from braintree.resource import Resource as Resource
from braintree.resource_collection import ResourceCollection as ResourceCollection
from braintree.subscription import Subscription as Subscription
from braintree.successful_result import SuccessfulResult as SuccessfulResult
from braintree.transaction import Transaction as Transaction

class SubscriptionGateway:
    gateway: Any
    config: Any
    def __init__(self, gateway) -> None: ...
    def cancel(self, subscription_id): ...
    def create(self, params: Incomplete | None = None): ...
    def find(self, subscription_id): ...
    def retry_charge(self, subscription_id, amount: Incomplete | None = None, submit_for_settlement: bool = False): ...
    def search(self, *query): ...
    def update(self, subscription_id, params: Incomplete | None = None): ...
