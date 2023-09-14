from _typeshed import Incomplete

from influxdb_client.service._base_service import _BaseService

class TasksService(_BaseService):
    def __init__(self, api_client: Incomplete | None = None) -> None: ...
    def delete_tasks_id(self, task_id, **kwargs): ...
    def delete_tasks_id_with_http_info(self, task_id, **kwargs): ...
    async def delete_tasks_id_async(self, task_id, **kwargs): ...
    def delete_tasks_id_labels_id(self, task_id, label_id, **kwargs): ...
    def delete_tasks_id_labels_id_with_http_info(self, task_id, label_id, **kwargs): ...
    async def delete_tasks_id_labels_id_async(self, task_id, label_id, **kwargs): ...
    def delete_tasks_id_members_id(self, user_id, task_id, **kwargs): ...
    def delete_tasks_id_members_id_with_http_info(self, user_id, task_id, **kwargs): ...
    async def delete_tasks_id_members_id_async(self, user_id, task_id, **kwargs): ...
    def delete_tasks_id_owners_id(self, user_id, task_id, **kwargs): ...
    def delete_tasks_id_owners_id_with_http_info(self, user_id, task_id, **kwargs): ...
    async def delete_tasks_id_owners_id_async(self, user_id, task_id, **kwargs): ...
    def delete_tasks_id_runs_id(self, task_id, run_id, **kwargs): ...
    def delete_tasks_id_runs_id_with_http_info(self, task_id, run_id, **kwargs): ...
    async def delete_tasks_id_runs_id_async(self, task_id, run_id, **kwargs): ...
    def get_tasks(self, **kwargs): ...
    def get_tasks_with_http_info(self, **kwargs): ...
    async def get_tasks_async(self, **kwargs): ...
    def get_tasks_id(self, task_id, **kwargs): ...
    def get_tasks_id_with_http_info(self, task_id, **kwargs): ...
    async def get_tasks_id_async(self, task_id, **kwargs): ...
    def get_tasks_id_labels(self, task_id, **kwargs): ...
    def get_tasks_id_labels_with_http_info(self, task_id, **kwargs): ...
    async def get_tasks_id_labels_async(self, task_id, **kwargs): ...
    def get_tasks_id_logs(self, task_id, **kwargs): ...
    def get_tasks_id_logs_with_http_info(self, task_id, **kwargs): ...
    async def get_tasks_id_logs_async(self, task_id, **kwargs): ...
    def get_tasks_id_members(self, task_id, **kwargs): ...
    def get_tasks_id_members_with_http_info(self, task_id, **kwargs): ...
    async def get_tasks_id_members_async(self, task_id, **kwargs): ...
    def get_tasks_id_owners(self, task_id, **kwargs): ...
    def get_tasks_id_owners_with_http_info(self, task_id, **kwargs): ...
    async def get_tasks_id_owners_async(self, task_id, **kwargs): ...
    def get_tasks_id_runs(self, task_id, **kwargs): ...
    def get_tasks_id_runs_with_http_info(self, task_id, **kwargs): ...
    async def get_tasks_id_runs_async(self, task_id, **kwargs): ...
    def get_tasks_id_runs_id(self, task_id, run_id, **kwargs): ...
    def get_tasks_id_runs_id_with_http_info(self, task_id, run_id, **kwargs): ...
    async def get_tasks_id_runs_id_async(self, task_id, run_id, **kwargs): ...
    def get_tasks_id_runs_id_logs(self, task_id, run_id, **kwargs): ...
    def get_tasks_id_runs_id_logs_with_http_info(self, task_id, run_id, **kwargs): ...
    async def get_tasks_id_runs_id_logs_async(self, task_id, run_id, **kwargs): ...
    def patch_tasks_id(self, task_id, task_update_request, **kwargs): ...
    def patch_tasks_id_with_http_info(self, task_id, task_update_request, **kwargs): ...
    async def patch_tasks_id_async(self, task_id, task_update_request, **kwargs): ...
    def post_tasks(self, task_create_request, **kwargs): ...
    def post_tasks_with_http_info(self, task_create_request, **kwargs): ...
    async def post_tasks_async(self, task_create_request, **kwargs): ...
    def post_tasks_id_labels(self, task_id, label_mapping, **kwargs): ...
    def post_tasks_id_labels_with_http_info(self, task_id, label_mapping, **kwargs): ...
    async def post_tasks_id_labels_async(self, task_id, label_mapping, **kwargs): ...
    def post_tasks_id_members(self, task_id, add_resource_member_request_body, **kwargs): ...
    def post_tasks_id_members_with_http_info(self, task_id, add_resource_member_request_body, **kwargs): ...
    async def post_tasks_id_members_async(self, task_id, add_resource_member_request_body, **kwargs): ...
    def post_tasks_id_owners(self, task_id, add_resource_member_request_body, **kwargs): ...
    def post_tasks_id_owners_with_http_info(self, task_id, add_resource_member_request_body, **kwargs): ...
    async def post_tasks_id_owners_async(self, task_id, add_resource_member_request_body, **kwargs): ...
    def post_tasks_id_runs(self, task_id, **kwargs): ...
    def post_tasks_id_runs_with_http_info(self, task_id, **kwargs): ...
    async def post_tasks_id_runs_async(self, task_id, **kwargs): ...
    def post_tasks_id_runs_id_retry(self, task_id, run_id, **kwargs): ...
    def post_tasks_id_runs_id_retry_with_http_info(self, task_id, run_id, **kwargs): ...
    async def post_tasks_id_runs_id_retry_async(self, task_id, run_id, **kwargs): ...
