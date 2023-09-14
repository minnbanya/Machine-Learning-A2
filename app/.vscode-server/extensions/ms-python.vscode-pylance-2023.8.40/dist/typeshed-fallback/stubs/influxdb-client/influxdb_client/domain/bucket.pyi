from _typeshed import Incomplete

class Bucket:
    openapi_types: Incomplete
    attribute_map: Incomplete
    discriminator: Incomplete
    def __init__(
        self,
        links: Incomplete | None = None,
        id: Incomplete | None = None,
        type: str = "user",
        name: Incomplete | None = None,
        description: Incomplete | None = None,
        org_id: Incomplete | None = None,
        rp: Incomplete | None = None,
        schema_type: Incomplete | None = None,
        created_at: Incomplete | None = None,
        updated_at: Incomplete | None = None,
        retention_rules: Incomplete | None = None,
        labels: Incomplete | None = None,
    ) -> None: ...
    @property
    def links(self): ...
    @links.setter
    def links(self, links) -> None: ...
    @property
    def id(self): ...
    @id.setter
    def id(self, id) -> None: ...
    @property
    def type(self): ...
    @type.setter
    def type(self, type) -> None: ...
    @property
    def name(self): ...
    @name.setter
    def name(self, name) -> None: ...
    @property
    def description(self): ...
    @description.setter
    def description(self, description) -> None: ...
    @property
    def org_id(self): ...
    @org_id.setter
    def org_id(self, org_id) -> None: ...
    @property
    def rp(self): ...
    @rp.setter
    def rp(self, rp) -> None: ...
    @property
    def schema_type(self): ...
    @schema_type.setter
    def schema_type(self, schema_type) -> None: ...
    @property
    def created_at(self): ...
    @created_at.setter
    def created_at(self, created_at) -> None: ...
    @property
    def updated_at(self): ...
    @updated_at.setter
    def updated_at(self, updated_at) -> None: ...
    @property
    def retention_rules(self): ...
    @retention_rules.setter
    def retention_rules(self, retention_rules) -> None: ...
    @property
    def labels(self): ...
    @labels.setter
    def labels(self, labels) -> None: ...
    def to_dict(self): ...
    def to_str(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
