import collections.abc
from _typeshed import Incomplete
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any, NoReturn

from parsimonious.expressions import _CALLABLE_TYPE, Expression, Literal, Lookahead, OneOf, Regex, Sequence, TokenMatcher
from parsimonious.nodes import Node, NodeVisitor

class Grammar(OrderedDict[str, Expression]):
    default_rule: Expression | Incomplete
    def __init__(self, rules: str = "", **more_rules: Expression | _CALLABLE_TYPE) -> None: ...
    def default(self, rule_name: str) -> Grammar: ...
    def parse(self, text: str, pos: int = 0) -> Node: ...
    def match(self, text: str, pos: int = 0) -> Node: ...

class TokenGrammar(Grammar): ...
class BootstrappingGrammar(Grammar): ...

rule_syntax: str

class LazyReference(str):
    name: str
    def resolve_refs(self, rule_map: Mapping[str, Expression | LazyReference]) -> Expression: ...

class RuleVisitor(NodeVisitor[tuple[OrderedDict[str, Expression], Expression | None]]):
    quantifier_classes: dict[str, type[Expression]]
    visit_expression: Callable[[RuleVisitor, Node, collections.abc.Sequence[Any]], Any]
    visit_term: Callable[[RuleVisitor, Node, collections.abc.Sequence[Any]], Any]
    visit_atom: Callable[[RuleVisitor, Node, collections.abc.Sequence[Any]], Any]
    custom_rules: dict[str, Expression]
    def __init__(self, custom_rules: Mapping[str, Expression] | None = None) -> None: ...
    def visit_parenthesized(self, node: Node, parenthesized: collections.abc.Sequence[Any]) -> Expression: ...
    def visit_quantifier(self, node: Node, quantifier: collections.abc.Sequence[Any]) -> Node: ...
    def visit_quantified(self, node: Node, quantified: collections.abc.Sequence[Any]) -> Expression: ...
    def visit_lookahead_term(self, node: Node, lookahead_term: collections.abc.Sequence[Any]) -> Lookahead: ...
    def visit_not_term(self, node: Node, not_term: collections.abc.Sequence[Any]) -> Lookahead: ...
    def visit_rule(self, node: Node, rule: collections.abc.Sequence[Any]) -> Expression: ...
    def visit_sequence(self, node: Node, sequence: collections.abc.Sequence[Any]) -> Sequence: ...
    def visit_ored(self, node: Node, ored: collections.abc.Sequence[Any]) -> OneOf: ...
    def visit_or_term(self, node: Node, or_term: collections.abc.Sequence[Any]) -> Expression: ...
    def visit_label(self, node: Node, label: collections.abc.Sequence[Any]) -> str: ...
    def visit_reference(self, node: Node, reference: collections.abc.Sequence[Any]) -> LazyReference: ...
    def visit_regex(self, node: Node, regex: collections.abc.Sequence[Any]) -> Regex: ...
    def visit_spaceless_literal(self, spaceless_literal: Node, visited_children: collections.abc.Sequence[Any]) -> Literal: ...
    def visit_literal(self, node: Node, literal: collections.abc.Sequence[Any]) -> Literal: ...
    def generic_visit(self, node: Node, visited_children: collections.abc.Sequence[Any]) -> collections.abc.Sequence[Any] | Node: ...  # type: ignore[override]
    def visit_rules(
        self, node: Node, rules_list: collections.abc.Sequence[Any]
    ) -> tuple[OrderedDict[str, Expression], Expression | None]: ...

class TokenRuleVisitor(RuleVisitor):
    def visit_spaceless_literal(
        self, spaceless_literal: Node, visited_children: collections.abc.Sequence[Any]
    ) -> TokenMatcher: ...
    def visit_regex(self, node: Node, regex: collections.abc.Sequence[Any]) -> NoReturn: ...

rule_grammar: Grammar
