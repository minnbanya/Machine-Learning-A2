from typing import Any

log: Any

class RequestValidator:
    def client_authentication_required(self, request, *args, **kwargs): ...
    def authenticate_client(self, request, *args, **kwargs) -> None: ...
    def authenticate_client_id(self, client_id, request, *args, **kwargs) -> None: ...
    def confirm_redirect_uri(self, client_id, code, redirect_uri, client, request, *args, **kwargs) -> None: ...
    def get_default_redirect_uri(self, client_id, request, *args, **kwargs) -> None: ...
    def get_default_scopes(self, client_id, request, *args, **kwargs) -> None: ...
    def get_original_scopes(self, refresh_token, request, *args, **kwargs) -> None: ...
    def is_within_original_scope(self, request_scopes, refresh_token, request, *args, **kwargs): ...
    def introspect_token(self, token, token_type_hint, request, *args, **kwargs) -> None: ...
    def invalidate_authorization_code(self, client_id, code, request, *args, **kwargs) -> None: ...
    def revoke_token(self, token, token_type_hint, request, *args, **kwargs) -> None: ...
    def rotate_refresh_token(self, request): ...
    def save_authorization_code(self, client_id, code, request, *args, **kwargs) -> None: ...
    def save_token(self, token, request, *args, **kwargs): ...
    def save_bearer_token(self, token, request, *args, **kwargs) -> None: ...
    def validate_bearer_token(self, token, scopes, request) -> None: ...
    def validate_client_id(self, client_id, request, *args, **kwargs) -> None: ...
    def validate_code(self, client_id, code, client, request, *args, **kwargs) -> None: ...
    def validate_grant_type(self, client_id, grant_type, client, request, *args, **kwargs) -> None: ...
    def validate_redirect_uri(self, client_id, redirect_uri, request, *args, **kwargs) -> None: ...
    def validate_refresh_token(self, refresh_token, client, request, *args, **kwargs) -> None: ...
    def validate_response_type(self, client_id, response_type, client, request, *args, **kwargs) -> None: ...
    def validate_scopes(self, client_id, scopes, client, request, *args, **kwargs) -> None: ...
    def validate_user(self, username, password, client, request, *args, **kwargs) -> None: ...
    def is_pkce_required(self, client_id, request): ...
    def get_code_challenge(self, code, request) -> None: ...
    def get_code_challenge_method(self, code, request) -> None: ...
