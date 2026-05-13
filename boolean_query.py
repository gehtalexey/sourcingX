"""Boolean query parser for Filter+ keyword inputs.

Supports the LinkedIn Sales Navigator syntax recruiters already know:
    (python OR java) AND senior NOT (junior OR intern)

Operators (case-insensitive):
    AND, OR, NOT — NOT is unary.
Grouping: ( )
Phrases: "machine learning" — matched as a literal substring.

Adjacent terms with no operator imply AND, so:
    python aws kubernetes
is equivalent to:
    python AND aws AND kubernetes

Term matching is case-insensitive substring against the target text. Empty
or whitespace-only queries match everything. Parse errors are swallowed
into match=True so a typo in the search box never wipes the result set.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Node:
    op: str                              # 'TERM' | 'AND' | 'OR' | 'NOT'
    children: List["Node"] = field(default_factory=list)
    value: Optional[str] = None          # populated only for TERM


# ---------- tokenizer ----------

def tokenize(query: str):
    """Split the query string into (kind, value) tokens."""
    tokens = []
    i = 0
    n = len(query)
    while i < n:
        c = query[i]
        if c.isspace():
            i += 1
        elif c == '(':
            tokens.append(('LPAREN', '('))
            i += 1
        elif c == ')':
            tokens.append(('RPAREN', ')'))
            i += 1
        elif c == '"':
            j = i + 1
            while j < n and query[j] != '"':
                j += 1
            phrase = query[i + 1:j]
            tokens.append(('TERM', phrase))
            i = j + 1 if j < n else j  # tolerate missing closing quote
        else:
            j = i
            while j < n and not query[j].isspace() and query[j] not in '()"':
                j += 1
            word = query[i:j]
            up = word.upper()
            if up == 'AND':
                tokens.append(('AND', 'AND'))
            elif up == 'OR':
                tokens.append(('OR', 'OR'))
            elif up == 'NOT':
                tokens.append(('NOT', 'NOT'))
            elif word:
                tokens.append(('TERM', word))
            i = j
    return tokens


# ---------- recursive-descent parser ----------

class _Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self):
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def parse_or(self):
        node = self.parse_and()
        while self.peek() and self.peek()[0] == 'OR':
            self.consume()
            right = self.parse_and()
            node = Node('OR', [node, right])
        return node

    def parse_and(self):
        node = self.parse_not()
        while True:
            t = self.peek()
            if t is None:
                break
            if t[0] == 'AND':
                self.consume()
                right = self.parse_not()
                node = Node('AND', [node, right])
            elif t[0] in ('TERM', 'NOT', 'LPAREN'):
                right = self.parse_not()
                node = Node('AND', [node, right])
            else:
                break
        return node

    def parse_not(self):
        if self.peek() and self.peek()[0] == 'NOT':
            self.consume()
            return Node('NOT', [self.parse_not()])
        return self.parse_atom()

    def parse_atom(self):
        t = self.peek()
        if t is None:
            return None
        if t[0] == 'LPAREN':
            self.consume()
            inner = self.parse_or()
            if self.peek() and self.peek()[0] == 'RPAREN':
                self.consume()
            return inner
        if t[0] == 'TERM':
            self.consume()
            return Node('TERM', value=t[1])
        # Stray operator at this position — skip it.
        self.consume()
        return None


def parse_query(query: str) -> Optional[Node]:
    tokens = tokenize(query)
    if not tokens:
        return None
    return _Parser(tokens).parse_or()


# ---------- evaluator ----------

def _eval(node: Optional[Node], text_lower: str) -> bool:
    if node is None:
        return True
    if node.op == 'TERM':
        return (node.value or '').lower() in text_lower
    if node.op == 'AND':
        return all(_eval(c, text_lower) for c in node.children)
    if node.op == 'OR':
        return any(_eval(c, text_lower) for c in node.children)
    if node.op == 'NOT':
        return not _eval(node.children[0], text_lower) if node.children else True
    return True


def match_boolean_query(query: str, text: str) -> bool:
    """Return True if `text` matches the boolean `query`.

    - Empty / whitespace-only query → True (no filter).
    - Parse errors → True (don't punish typos).
    - Term match is case-insensitive substring.
    """
    if not query or not query.strip():
        return True
    text_lower = (text or '').lower()
    try:
        return _eval(parse_query(query), text_lower)
    except Exception:
        return True
