"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import s2clientprotocol.common_pb2
import sys
import typing

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class _DisplayType:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _DisplayTypeEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_DisplayType.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    Visible: _DisplayType.ValueType  # 1
    """Fully visible"""
    Snapshot: _DisplayType.ValueType  # 2
    """Dimmed version of unit left behind after entering fog of war"""
    Hidden: _DisplayType.ValueType  # 3
    """Fully hidden"""
    Placeholder: _DisplayType.ValueType  # 4
    """Building that hasn't started construction."""

class DisplayType(_DisplayType, metaclass=_DisplayTypeEnumTypeWrapper): ...

Visible: DisplayType.ValueType  # 1
"""Fully visible"""
Snapshot: DisplayType.ValueType  # 2
"""Dimmed version of unit left behind after entering fog of war"""
Hidden: DisplayType.ValueType  # 3
"""Fully hidden"""
Placeholder: DisplayType.ValueType  # 4
"""Building that hasn't started construction."""
global___DisplayType = DisplayType

class _Alliance:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _AllianceEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_Alliance.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    Self: _Alliance.ValueType  # 1
    Ally: _Alliance.ValueType  # 2
    Neutral: _Alliance.ValueType  # 3
    Enemy: _Alliance.ValueType  # 4

class Alliance(_Alliance, metaclass=_AllianceEnumTypeWrapper): ...

Self: Alliance.ValueType  # 1
Ally: Alliance.ValueType  # 2
Neutral: Alliance.ValueType  # 3
Enemy: Alliance.ValueType  # 4
global___Alliance = Alliance

class _CloakState:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _CloakStateEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_CloakState.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    CloakedUnknown: _CloakState.ValueType  # 0
    """Under the fog, so unknown whether it's cloaked or not."""
    Cloaked: _CloakState.ValueType  # 1
    CloakedDetected: _CloakState.ValueType  # 2
    NotCloaked: _CloakState.ValueType  # 3
    CloakedAllied: _CloakState.ValueType  # 4

class CloakState(_CloakState, metaclass=_CloakStateEnumTypeWrapper): ...

CloakedUnknown: CloakState.ValueType  # 0
"""Under the fog, so unknown whether it's cloaked or not."""
Cloaked: CloakState.ValueType  # 1
CloakedDetected: CloakState.ValueType  # 2
NotCloaked: CloakState.ValueType  # 3
CloakedAllied: CloakState.ValueType  # 4
global___CloakState = CloakState

@typing_extensions.final
class StartRaw(google.protobuf.message.Message):
    """
    Start
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    MAP_SIZE_FIELD_NUMBER: builtins.int
    PATHING_GRID_FIELD_NUMBER: builtins.int
    TERRAIN_HEIGHT_FIELD_NUMBER: builtins.int
    PLACEMENT_GRID_FIELD_NUMBER: builtins.int
    PLAYABLE_AREA_FIELD_NUMBER: builtins.int
    START_LOCATIONS_FIELD_NUMBER: builtins.int
    @property
    def map_size(self) -> s2clientprotocol.common_pb2.Size2DI:
        """Width and height of the map."""
    @property
    def pathing_grid(self) -> s2clientprotocol.common_pb2.ImageData:
        """1 bit bitmap of the pathing grid."""
    @property
    def terrain_height(self) -> s2clientprotocol.common_pb2.ImageData:
        """1 byte bitmap of the terrain height."""
    @property
    def placement_grid(self) -> s2clientprotocol.common_pb2.ImageData:
        """1 bit bitmap of the building placement grid."""
    @property
    def playable_area(self) -> s2clientprotocol.common_pb2.RectangleI:
        """The playable cells."""
    @property
    def start_locations(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[s2clientprotocol.common_pb2.Point2D]:
        """Possible start locations for players."""
    def __init__(
        self,
        *,
        map_size: s2clientprotocol.common_pb2.Size2DI | None = ...,
        pathing_grid: s2clientprotocol.common_pb2.ImageData | None = ...,
        terrain_height: s2clientprotocol.common_pb2.ImageData | None = ...,
        placement_grid: s2clientprotocol.common_pb2.ImageData | None = ...,
        playable_area: s2clientprotocol.common_pb2.RectangleI | None = ...,
        start_locations: collections.abc.Iterable[s2clientprotocol.common_pb2.Point2D] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["map_size", b"map_size", "pathing_grid", b"pathing_grid", "placement_grid", b"placement_grid", "playable_area", b"playable_area", "terrain_height", b"terrain_height"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["map_size", b"map_size", "pathing_grid", b"pathing_grid", "placement_grid", b"placement_grid", "playable_area", b"playable_area", "start_locations", b"start_locations", "terrain_height", b"terrain_height"]) -> None: ...

global___StartRaw = StartRaw

@typing_extensions.final
class ObservationRaw(google.protobuf.message.Message):
    """
    Observation
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    PLAYER_FIELD_NUMBER: builtins.int
    UNITS_FIELD_NUMBER: builtins.int
    MAP_STATE_FIELD_NUMBER: builtins.int
    EVENT_FIELD_NUMBER: builtins.int
    EFFECTS_FIELD_NUMBER: builtins.int
    RADAR_FIELD_NUMBER: builtins.int
    @property
    def player(self) -> global___PlayerRaw: ...
    @property
    def units(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Unit]: ...
    @property
    def map_state(self) -> global___MapState:
        """Fog of war, creep and so on. Board stuff that changes per frame."""
    @property
    def event(self) -> global___Event: ...
    @property
    def effects(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Effect]: ...
    @property
    def radar(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___RadarRing]: ...
    def __init__(
        self,
        *,
        player: global___PlayerRaw | None = ...,
        units: collections.abc.Iterable[global___Unit] | None = ...,
        map_state: global___MapState | None = ...,
        event: global___Event | None = ...,
        effects: collections.abc.Iterable[global___Effect] | None = ...,
        radar: collections.abc.Iterable[global___RadarRing] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["event", b"event", "map_state", b"map_state", "player", b"player"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["effects", b"effects", "event", b"event", "map_state", b"map_state", "player", b"player", "radar", b"radar", "units", b"units"]) -> None: ...

global___ObservationRaw = ObservationRaw

@typing_extensions.final
class RadarRing(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    POS_FIELD_NUMBER: builtins.int
    RADIUS_FIELD_NUMBER: builtins.int
    @property
    def pos(self) -> s2clientprotocol.common_pb2.Point: ...
    radius: builtins.float
    def __init__(
        self,
        *,
        pos: s2clientprotocol.common_pb2.Point | None = ...,
        radius: builtins.float | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["pos", b"pos", "radius", b"radius"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["pos", b"pos", "radius", b"radius"]) -> None: ...

global___RadarRing = RadarRing

@typing_extensions.final
class PowerSource(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    POS_FIELD_NUMBER: builtins.int
    RADIUS_FIELD_NUMBER: builtins.int
    TAG_FIELD_NUMBER: builtins.int
    @property
    def pos(self) -> s2clientprotocol.common_pb2.Point: ...
    radius: builtins.float
    tag: builtins.int
    def __init__(
        self,
        *,
        pos: s2clientprotocol.common_pb2.Point | None = ...,
        radius: builtins.float | None = ...,
        tag: builtins.int | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["pos", b"pos", "radius", b"radius", "tag", b"tag"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["pos", b"pos", "radius", b"radius", "tag", b"tag"]) -> None: ...

global___PowerSource = PowerSource

@typing_extensions.final
class PlayerRaw(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    POWER_SOURCES_FIELD_NUMBER: builtins.int
    CAMERA_FIELD_NUMBER: builtins.int
    UPGRADE_IDS_FIELD_NUMBER: builtins.int
    @property
    def power_sources(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___PowerSource]: ...
    @property
    def camera(self) -> s2clientprotocol.common_pb2.Point: ...
    @property
    def upgrade_ids(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """TODO: Add to UI observation?"""
    def __init__(
        self,
        *,
        power_sources: collections.abc.Iterable[global___PowerSource] | None = ...,
        camera: s2clientprotocol.common_pb2.Point | None = ...,
        upgrade_ids: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["camera", b"camera"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["camera", b"camera", "power_sources", b"power_sources", "upgrade_ids", b"upgrade_ids"]) -> None: ...

global___PlayerRaw = PlayerRaw

@typing_extensions.final
class UnitOrder(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ABILITY_ID_FIELD_NUMBER: builtins.int
    TARGET_WORLD_SPACE_POS_FIELD_NUMBER: builtins.int
    TARGET_UNIT_TAG_FIELD_NUMBER: builtins.int
    PROGRESS_FIELD_NUMBER: builtins.int
    ability_id: builtins.int
    @property
    def target_world_space_pos(self) -> s2clientprotocol.common_pb2.Point: ...
    target_unit_tag: builtins.int
    progress: builtins.float
    """Progress of train abilities. Range: [0.0, 1.0]"""
    def __init__(
        self,
        *,
        ability_id: builtins.int | None = ...,
        target_world_space_pos: s2clientprotocol.common_pb2.Point | None = ...,
        target_unit_tag: builtins.int | None = ...,
        progress: builtins.float | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["ability_id", b"ability_id", "progress", b"progress", "target", b"target", "target_unit_tag", b"target_unit_tag", "target_world_space_pos", b"target_world_space_pos"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["ability_id", b"ability_id", "progress", b"progress", "target", b"target", "target_unit_tag", b"target_unit_tag", "target_world_space_pos", b"target_world_space_pos"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["target", b"target"]) -> typing_extensions.Literal["target_world_space_pos", "target_unit_tag"] | None: ...

global___UnitOrder = UnitOrder

@typing_extensions.final
class PassengerUnit(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TAG_FIELD_NUMBER: builtins.int
    HEALTH_FIELD_NUMBER: builtins.int
    HEALTH_MAX_FIELD_NUMBER: builtins.int
    SHIELD_FIELD_NUMBER: builtins.int
    SHIELD_MAX_FIELD_NUMBER: builtins.int
    ENERGY_FIELD_NUMBER: builtins.int
    ENERGY_MAX_FIELD_NUMBER: builtins.int
    UNIT_TYPE_FIELD_NUMBER: builtins.int
    tag: builtins.int
    health: builtins.float
    health_max: builtins.float
    shield: builtins.float
    shield_max: builtins.float
    energy: builtins.float
    energy_max: builtins.float
    unit_type: builtins.int
    def __init__(
        self,
        *,
        tag: builtins.int | None = ...,
        health: builtins.float | None = ...,
        health_max: builtins.float | None = ...,
        shield: builtins.float | None = ...,
        shield_max: builtins.float | None = ...,
        energy: builtins.float | None = ...,
        energy_max: builtins.float | None = ...,
        unit_type: builtins.int | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["energy", b"energy", "energy_max", b"energy_max", "health", b"health", "health_max", b"health_max", "shield", b"shield", "shield_max", b"shield_max", "tag", b"tag", "unit_type", b"unit_type"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["energy", b"energy", "energy_max", b"energy_max", "health", b"health", "health_max", b"health_max", "shield", b"shield", "shield_max", b"shield_max", "tag", b"tag", "unit_type", b"unit_type"]) -> None: ...

global___PassengerUnit = PassengerUnit

@typing_extensions.final
class RallyTarget(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    POINT_FIELD_NUMBER: builtins.int
    TAG_FIELD_NUMBER: builtins.int
    @property
    def point(self) -> s2clientprotocol.common_pb2.Point:
        """Will always be filled."""
    tag: builtins.int
    """Only if it's targeting a unit."""
    def __init__(
        self,
        *,
        point: s2clientprotocol.common_pb2.Point | None = ...,
        tag: builtins.int | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["point", b"point", "tag", b"tag"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["point", b"point", "tag", b"tag"]) -> None: ...

global___RallyTarget = RallyTarget

@typing_extensions.final
class Unit(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DISPLAY_TYPE_FIELD_NUMBER: builtins.int
    ALLIANCE_FIELD_NUMBER: builtins.int
    TAG_FIELD_NUMBER: builtins.int
    UNIT_TYPE_FIELD_NUMBER: builtins.int
    OWNER_FIELD_NUMBER: builtins.int
    POS_FIELD_NUMBER: builtins.int
    FACING_FIELD_NUMBER: builtins.int
    RADIUS_FIELD_NUMBER: builtins.int
    BUILD_PROGRESS_FIELD_NUMBER: builtins.int
    CLOAK_FIELD_NUMBER: builtins.int
    BUFF_IDS_FIELD_NUMBER: builtins.int
    DETECT_RANGE_FIELD_NUMBER: builtins.int
    RADAR_RANGE_FIELD_NUMBER: builtins.int
    IS_SELECTED_FIELD_NUMBER: builtins.int
    IS_ON_SCREEN_FIELD_NUMBER: builtins.int
    IS_BLIP_FIELD_NUMBER: builtins.int
    IS_POWERED_FIELD_NUMBER: builtins.int
    IS_ACTIVE_FIELD_NUMBER: builtins.int
    ATTACK_UPGRADE_LEVEL_FIELD_NUMBER: builtins.int
    ARMOR_UPGRADE_LEVEL_FIELD_NUMBER: builtins.int
    SHIELD_UPGRADE_LEVEL_FIELD_NUMBER: builtins.int
    HEALTH_FIELD_NUMBER: builtins.int
    HEALTH_MAX_FIELD_NUMBER: builtins.int
    SHIELD_FIELD_NUMBER: builtins.int
    SHIELD_MAX_FIELD_NUMBER: builtins.int
    ENERGY_FIELD_NUMBER: builtins.int
    ENERGY_MAX_FIELD_NUMBER: builtins.int
    MINERAL_CONTENTS_FIELD_NUMBER: builtins.int
    VESPENE_CONTENTS_FIELD_NUMBER: builtins.int
    IS_FLYING_FIELD_NUMBER: builtins.int
    IS_BURROWED_FIELD_NUMBER: builtins.int
    IS_HALLUCINATION_FIELD_NUMBER: builtins.int
    ORDERS_FIELD_NUMBER: builtins.int
    ADD_ON_TAG_FIELD_NUMBER: builtins.int
    PASSENGERS_FIELD_NUMBER: builtins.int
    CARGO_SPACE_TAKEN_FIELD_NUMBER: builtins.int
    CARGO_SPACE_MAX_FIELD_NUMBER: builtins.int
    ASSIGNED_HARVESTERS_FIELD_NUMBER: builtins.int
    IDEAL_HARVESTERS_FIELD_NUMBER: builtins.int
    WEAPON_COOLDOWN_FIELD_NUMBER: builtins.int
    ENGAGED_TARGET_TAG_FIELD_NUMBER: builtins.int
    BUFF_DURATION_REMAIN_FIELD_NUMBER: builtins.int
    BUFF_DURATION_MAX_FIELD_NUMBER: builtins.int
    RALLY_TARGETS_FIELD_NUMBER: builtins.int
    display_type: global___DisplayType.ValueType
    """Fields are populated based on type/alliance"""
    alliance: global___Alliance.ValueType
    tag: builtins.int
    """Unique identifier for a unit"""
    unit_type: builtins.int
    owner: builtins.int
    @property
    def pos(self) -> s2clientprotocol.common_pb2.Point: ...
    facing: builtins.float
    radius: builtins.float
    build_progress: builtins.float
    """Range: [0.0, 1.0]"""
    cloak: global___CloakState.ValueType
    @property
    def buff_ids(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    detect_range: builtins.float
    radar_range: builtins.float
    is_selected: builtins.bool
    is_on_screen: builtins.bool
    """Visible and within the camera frustrum."""
    is_blip: builtins.bool
    """Detected by sensor tower"""
    is_powered: builtins.bool
    is_active: builtins.bool
    """Building is training/researching (ie animated)."""
    attack_upgrade_level: builtins.int
    armor_upgrade_level: builtins.int
    shield_upgrade_level: builtins.int
    health: builtins.float
    """Not populated for snapshots"""
    health_max: builtins.float
    shield: builtins.float
    shield_max: builtins.float
    energy: builtins.float
    energy_max: builtins.float
    mineral_contents: builtins.int
    vespene_contents: builtins.int
    is_flying: builtins.bool
    is_burrowed: builtins.bool
    is_hallucination: builtins.bool
    """Unit is your own or detected as a hallucination."""
    @property
    def orders(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___UnitOrder]:
        """Not populated for enemies"""
    add_on_tag: builtins.int
    @property
    def passengers(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___PassengerUnit]: ...
    cargo_space_taken: builtins.int
    cargo_space_max: builtins.int
    assigned_harvesters: builtins.int
    ideal_harvesters: builtins.int
    weapon_cooldown: builtins.float
    engaged_target_tag: builtins.int
    buff_duration_remain: builtins.int
    """How long a buff or unit is still around (eg mule, broodling, chronoboost)."""
    buff_duration_max: builtins.int
    """How long the buff or unit is still around (eg mule, broodling, chronoboost)."""
    @property
    def rally_targets(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___RallyTarget]: ...
    def __init__(
        self,
        *,
        display_type: global___DisplayType.ValueType | None = ...,
        alliance: global___Alliance.ValueType | None = ...,
        tag: builtins.int | None = ...,
        unit_type: builtins.int | None = ...,
        owner: builtins.int | None = ...,
        pos: s2clientprotocol.common_pb2.Point | None = ...,
        facing: builtins.float | None = ...,
        radius: builtins.float | None = ...,
        build_progress: builtins.float | None = ...,
        cloak: global___CloakState.ValueType | None = ...,
        buff_ids: collections.abc.Iterable[builtins.int] | None = ...,
        detect_range: builtins.float | None = ...,
        radar_range: builtins.float | None = ...,
        is_selected: builtins.bool | None = ...,
        is_on_screen: builtins.bool | None = ...,
        is_blip: builtins.bool | None = ...,
        is_powered: builtins.bool | None = ...,
        is_active: builtins.bool | None = ...,
        attack_upgrade_level: builtins.int | None = ...,
        armor_upgrade_level: builtins.int | None = ...,
        shield_upgrade_level: builtins.int | None = ...,
        health: builtins.float | None = ...,
        health_max: builtins.float | None = ...,
        shield: builtins.float | None = ...,
        shield_max: builtins.float | None = ...,
        energy: builtins.float | None = ...,
        energy_max: builtins.float | None = ...,
        mineral_contents: builtins.int | None = ...,
        vespene_contents: builtins.int | None = ...,
        is_flying: builtins.bool | None = ...,
        is_burrowed: builtins.bool | None = ...,
        is_hallucination: builtins.bool | None = ...,
        orders: collections.abc.Iterable[global___UnitOrder] | None = ...,
        add_on_tag: builtins.int | None = ...,
        passengers: collections.abc.Iterable[global___PassengerUnit] | None = ...,
        cargo_space_taken: builtins.int | None = ...,
        cargo_space_max: builtins.int | None = ...,
        assigned_harvesters: builtins.int | None = ...,
        ideal_harvesters: builtins.int | None = ...,
        weapon_cooldown: builtins.float | None = ...,
        engaged_target_tag: builtins.int | None = ...,
        buff_duration_remain: builtins.int | None = ...,
        buff_duration_max: builtins.int | None = ...,
        rally_targets: collections.abc.Iterable[global___RallyTarget] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["add_on_tag", b"add_on_tag", "alliance", b"alliance", "armor_upgrade_level", b"armor_upgrade_level", "assigned_harvesters", b"assigned_harvesters", "attack_upgrade_level", b"attack_upgrade_level", "buff_duration_max", b"buff_duration_max", "buff_duration_remain", b"buff_duration_remain", "build_progress", b"build_progress", "cargo_space_max", b"cargo_space_max", "cargo_space_taken", b"cargo_space_taken", "cloak", b"cloak", "detect_range", b"detect_range", "display_type", b"display_type", "energy", b"energy", "energy_max", b"energy_max", "engaged_target_tag", b"engaged_target_tag", "facing", b"facing", "health", b"health", "health_max", b"health_max", "ideal_harvesters", b"ideal_harvesters", "is_active", b"is_active", "is_blip", b"is_blip", "is_burrowed", b"is_burrowed", "is_flying", b"is_flying", "is_hallucination", b"is_hallucination", "is_on_screen", b"is_on_screen", "is_powered", b"is_powered", "is_selected", b"is_selected", "mineral_contents", b"mineral_contents", "owner", b"owner", "pos", b"pos", "radar_range", b"radar_range", "radius", b"radius", "shield", b"shield", "shield_max", b"shield_max", "shield_upgrade_level", b"shield_upgrade_level", "tag", b"tag", "unit_type", b"unit_type", "vespene_contents", b"vespene_contents", "weapon_cooldown", b"weapon_cooldown"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["add_on_tag", b"add_on_tag", "alliance", b"alliance", "armor_upgrade_level", b"armor_upgrade_level", "assigned_harvesters", b"assigned_harvesters", "attack_upgrade_level", b"attack_upgrade_level", "buff_duration_max", b"buff_duration_max", "buff_duration_remain", b"buff_duration_remain", "buff_ids", b"buff_ids", "build_progress", b"build_progress", "cargo_space_max", b"cargo_space_max", "cargo_space_taken", b"cargo_space_taken", "cloak", b"cloak", "detect_range", b"detect_range", "display_type", b"display_type", "energy", b"energy", "energy_max", b"energy_max", "engaged_target_tag", b"engaged_target_tag", "facing", b"facing", "health", b"health", "health_max", b"health_max", "ideal_harvesters", b"ideal_harvesters", "is_active", b"is_active", "is_blip", b"is_blip", "is_burrowed", b"is_burrowed", "is_flying", b"is_flying", "is_hallucination", b"is_hallucination", "is_on_screen", b"is_on_screen", "is_powered", b"is_powered", "is_selected", b"is_selected", "mineral_contents", b"mineral_contents", "orders", b"orders", "owner", b"owner", "passengers", b"passengers", "pos", b"pos", "radar_range", b"radar_range", "radius", b"radius", "rally_targets", b"rally_targets", "shield", b"shield", "shield_max", b"shield_max", "shield_upgrade_level", b"shield_upgrade_level", "tag", b"tag", "unit_type", b"unit_type", "vespene_contents", b"vespene_contents", "weapon_cooldown", b"weapon_cooldown"]) -> None: ...

global___Unit = Unit

@typing_extensions.final
class MapState(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VISIBILITY_FIELD_NUMBER: builtins.int
    CREEP_FIELD_NUMBER: builtins.int
    @property
    def visibility(self) -> s2clientprotocol.common_pb2.ImageData:
        """1 byte visibility layer."""
    @property
    def creep(self) -> s2clientprotocol.common_pb2.ImageData:
        """1 bit creep layer."""
    def __init__(
        self,
        *,
        visibility: s2clientprotocol.common_pb2.ImageData | None = ...,
        creep: s2clientprotocol.common_pb2.ImageData | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["creep", b"creep", "visibility", b"visibility"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["creep", b"creep", "visibility", b"visibility"]) -> None: ...

global___MapState = MapState

@typing_extensions.final
class Event(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DEAD_UNITS_FIELD_NUMBER: builtins.int
    @property
    def dead_units(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    def __init__(
        self,
        *,
        dead_units: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["dead_units", b"dead_units"]) -> None: ...

global___Event = Event

@typing_extensions.final
class Effect(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    EFFECT_ID_FIELD_NUMBER: builtins.int
    POS_FIELD_NUMBER: builtins.int
    ALLIANCE_FIELD_NUMBER: builtins.int
    OWNER_FIELD_NUMBER: builtins.int
    RADIUS_FIELD_NUMBER: builtins.int
    effect_id: builtins.int
    @property
    def pos(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[s2clientprotocol.common_pb2.Point2D]:
        """Effect may impact multiple locations. (eg. Lurker attack)"""
    alliance: global___Alliance.ValueType
    owner: builtins.int
    radius: builtins.float
    def __init__(
        self,
        *,
        effect_id: builtins.int | None = ...,
        pos: collections.abc.Iterable[s2clientprotocol.common_pb2.Point2D] | None = ...,
        alliance: global___Alliance.ValueType | None = ...,
        owner: builtins.int | None = ...,
        radius: builtins.float | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["alliance", b"alliance", "effect_id", b"effect_id", "owner", b"owner", "radius", b"radius"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["alliance", b"alliance", "effect_id", b"effect_id", "owner", b"owner", "pos", b"pos", "radius", b"radius"]) -> None: ...

global___Effect = Effect

@typing_extensions.final
class ActionRaw(google.protobuf.message.Message):
    """
    Action
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    UNIT_COMMAND_FIELD_NUMBER: builtins.int
    CAMERA_MOVE_FIELD_NUMBER: builtins.int
    TOGGLE_AUTOCAST_FIELD_NUMBER: builtins.int
    @property
    def unit_command(self) -> global___ActionRawUnitCommand: ...
    @property
    def camera_move(self) -> global___ActionRawCameraMove: ...
    @property
    def toggle_autocast(self) -> global___ActionRawToggleAutocast: ...
    def __init__(
        self,
        *,
        unit_command: global___ActionRawUnitCommand | None = ...,
        camera_move: global___ActionRawCameraMove | None = ...,
        toggle_autocast: global___ActionRawToggleAutocast | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["action", b"action", "camera_move", b"camera_move", "toggle_autocast", b"toggle_autocast", "unit_command", b"unit_command"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["action", b"action", "camera_move", b"camera_move", "toggle_autocast", b"toggle_autocast", "unit_command", b"unit_command"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["action", b"action"]) -> typing_extensions.Literal["unit_command", "camera_move", "toggle_autocast"] | None: ...

global___ActionRaw = ActionRaw

@typing_extensions.final
class ActionRawUnitCommand(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ABILITY_ID_FIELD_NUMBER: builtins.int
    TARGET_WORLD_SPACE_POS_FIELD_NUMBER: builtins.int
    TARGET_UNIT_TAG_FIELD_NUMBER: builtins.int
    UNIT_TAGS_FIELD_NUMBER: builtins.int
    QUEUE_COMMAND_FIELD_NUMBER: builtins.int
    ability_id: builtins.int
    @property
    def target_world_space_pos(self) -> s2clientprotocol.common_pb2.Point2D: ...
    target_unit_tag: builtins.int
    @property
    def unit_tags(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    queue_command: builtins.bool
    def __init__(
        self,
        *,
        ability_id: builtins.int | None = ...,
        target_world_space_pos: s2clientprotocol.common_pb2.Point2D | None = ...,
        target_unit_tag: builtins.int | None = ...,
        unit_tags: collections.abc.Iterable[builtins.int] | None = ...,
        queue_command: builtins.bool | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["ability_id", b"ability_id", "queue_command", b"queue_command", "target", b"target", "target_unit_tag", b"target_unit_tag", "target_world_space_pos", b"target_world_space_pos"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["ability_id", b"ability_id", "queue_command", b"queue_command", "target", b"target", "target_unit_tag", b"target_unit_tag", "target_world_space_pos", b"target_world_space_pos", "unit_tags", b"unit_tags"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["target", b"target"]) -> typing_extensions.Literal["target_world_space_pos", "target_unit_tag"] | None: ...

global___ActionRawUnitCommand = ActionRawUnitCommand

@typing_extensions.final
class ActionRawCameraMove(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    CENTER_WORLD_SPACE_FIELD_NUMBER: builtins.int
    @property
    def center_world_space(self) -> s2clientprotocol.common_pb2.Point: ...
    def __init__(
        self,
        *,
        center_world_space: s2clientprotocol.common_pb2.Point | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["center_world_space", b"center_world_space"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["center_world_space", b"center_world_space"]) -> None: ...

global___ActionRawCameraMove = ActionRawCameraMove

@typing_extensions.final
class ActionRawToggleAutocast(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ABILITY_ID_FIELD_NUMBER: builtins.int
    UNIT_TAGS_FIELD_NUMBER: builtins.int
    ability_id: builtins.int
    @property
    def unit_tags(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    def __init__(
        self,
        *,
        ability_id: builtins.int | None = ...,
        unit_tags: collections.abc.Iterable[builtins.int] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["ability_id", b"ability_id"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["ability_id", b"ability_id", "unit_tags", b"unit_tags"]) -> None: ...

global___ActionRawToggleAutocast = ActionRawToggleAutocast
