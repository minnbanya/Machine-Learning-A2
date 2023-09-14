from typing import Any

from braintree.error_result import ErrorResult as ErrorResult
from braintree.exceptions.not_found_error import NotFoundError as NotFoundError
from braintree.oauth_credentials import OAuthCredentials as OAuthCredentials
from braintree.successful_result import SuccessfulResult as SuccessfulResult

class OAuthGateway:
    gateway: Any
    config: Any
    def __init__(self, gateway) -> None: ...
    def create_token_from_code(self, params): ...
    def create_token_from_refresh_token(self, params): ...
    def revoke_access_token(self, access_token): ...
    def connect_url(self, raw_params): ...
