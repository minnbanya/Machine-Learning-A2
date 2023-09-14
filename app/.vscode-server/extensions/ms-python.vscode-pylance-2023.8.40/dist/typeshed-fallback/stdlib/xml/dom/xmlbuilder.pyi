from _typeshed import Incomplete, Unused
from typing import Any, NoReturn
from typing_extensions import Literal, TypeAlias
from urllib.request import OpenerDirector
from xml.dom.expatbuilder import ExpatBuilder, ExpatBuilderNS
from xml.dom.minidom import Node

__all__ = ["DOMBuilder", "DOMEntityResolver", "DOMInputSource"]

# UNKNOWN TYPES:
# - `Options.errorHandler`.
#       The same as `_DOMBuilderErrorHandlerType`?
#       Maybe `xml.sax.handler.ErrorHandler`?
# - Return type of DOMBuilder.getFeature().
#       We could get rid of the `Incomplete` if we knew more
#       about `Options.errorHandler`.

# ALIASES REPRESENTING MORE UNKNOWN TYPES:

# probably the same as `Options.errorHandler`?
# Maybe `xml.sax.handler.ErrorHandler`?
_DOMBuilderErrorHandlerType: TypeAlias = Incomplete | None
# probably some kind of IO...
_DOMInputSourceCharacterStreamType: TypeAlias = Incomplete | None
# probably a string??
_DOMInputSourceStringDataType: TypeAlias = Incomplete | None
# probably a string??
_DOMInputSourceEncodingType: TypeAlias = Incomplete | None

class Options:
    namespaces: int
    namespace_declarations: bool
    validation: bool
    external_parameter_entities: bool
    external_general_entities: bool
    external_dtd_subset: bool
    validate_if_schema: bool
    validate: bool
    datatype_normalization: bool
    create_entity_ref_nodes: bool
    entities: bool
    whitespace_in_element_content: bool
    cdata_sections: bool
    comments: bool
    charset_overrides_xml_encoding: bool
    infoset: bool
    supported_mediatypes_only: bool
    errorHandler: Any | None
    filter: DOMBuilderFilter | None  # a guess, but seems likely

class DOMBuilder:
    entityResolver: DOMEntityResolver | None  # a guess, but seems likely
    errorHandler: _DOMBuilderErrorHandlerType
    filter: DOMBuilderFilter | None  # a guess, but seems likely
    ACTION_REPLACE: Literal[1]
    ACTION_APPEND_AS_CHILDREN: Literal[2]
    ACTION_INSERT_AFTER: Literal[3]
    ACTION_INSERT_BEFORE: Literal[4]
    def setFeature(self, name: str, state: int) -> None: ...
    def supportsFeature(self, name: str) -> bool: ...
    def canSetFeature(self, name: str, state: int) -> bool: ...
    # getFeature could return any attribute from an instance of `Options`
    def getFeature(self, name: str) -> Incomplete: ...
    def parseURI(self, uri: str) -> ExpatBuilder | ExpatBuilderNS: ...
    def parse(self, input: DOMInputSource) -> ExpatBuilder | ExpatBuilderNS: ...
    # `input` and `cnode` argtypes for `parseWithContext` are unknowable
    # as the function does nothing with them, and always raises an exception.
    # But `input` is *probably* `DOMInputSource`?
    def parseWithContext(self, input: Unused, cnode: Unused, action: Literal[1, 2, 3, 4]) -> NoReturn: ...

class DOMEntityResolver:
    def resolveEntity(self, publicId: str | None, systemId: str) -> DOMInputSource: ...

class DOMInputSource:
    byteStream: OpenerDirector | None
    characterStream: _DOMInputSourceCharacterStreamType
    stringData: _DOMInputSourceStringDataType
    encoding: _DOMInputSourceEncodingType
    publicId: str | None
    systemId: str | None
    baseURI: str | None

class DOMBuilderFilter:
    FILTER_ACCEPT: Literal[1]
    FILTER_REJECT: Literal[2]
    FILTER_SKIP: Literal[3]
    FILTER_INTERRUPT: Literal[4]
    whatToShow: int
    def acceptNode(self, element: Unused) -> Literal[1]: ...
    def startContainer(self, element: Unused) -> Literal[1]: ...

class DocumentLS:
    async_: bool
    def abort(self) -> NoReturn: ...
    # `load()` and `loadXML()` always raise exceptions
    # so the argtypes of `uri` and `source` are unknowable.
    # `source` is *probably* `DOMInputSource`?
    # `uri` is *probably* a str? (see DOMBuilder.parseURI())
    def load(self, uri: Unused) -> NoReturn: ...
    def loadXML(self, source: Unused) -> NoReturn: ...
    def saveXML(self, snode: Node | None) -> str: ...

class DOMImplementationLS:
    MODE_SYNCHRONOUS: Literal[1]
    MODE_ASYNCHRONOUS: Literal[2]
    def createDOMBuilder(self, mode: Literal[1], schemaType: None) -> DOMBuilder: ...
    def createDOMWriter(self) -> NoReturn: ...
    def createDOMInputSource(self) -> DOMInputSource: ...
