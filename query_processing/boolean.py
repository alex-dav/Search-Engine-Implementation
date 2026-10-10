"""Boolean query processor for Task 1."""

from typing import List, Set

from index.access import get_all_doc_ids, get_posting_list


def _shunting_yard_reorder(tokens: List[str]) -> List[str]:
    """Reorder infix Boolean tokens into postfix form."""
    output_queue: List[str] = []
    operator_stack: List[str] = []

    # See the Week 04 lecture section "Shunting-Yard Algorithm". Follow the
    # provided pseudocode using NOT > AND > OR.
    # (Task 1): complete the conversion.
    for token in tokens:
        if token in {"AND", "OR", "NOT"}:
            while (operator_stack and operator_stack[-1] != "(" and
                   (token == "OR" or
                   (token == "AND" and operator_stack[-1] != "OR") or
                   (token == "NOT" and operator_stack[-1] not in {"AND", "OR"}))):
                output_queue.append(operator_stack.pop())
            operator_stack.append(token)
        elif token == "(":
            operator_stack.append(token)
        elif token == ")":
            while operator_stack and operator_stack[-1] != "(":
                output_queue.append(operator_stack.pop())
            if operator_stack and operator_stack[-1] == "(":
                operator_stack.pop()
        else:
            output_queue.append(token)
    while operator_stack:
        output_queue.append(operator_stack.pop())
    return output_queue


def _evaluate_postfix(postfix_tokens: List[str], index_path: str) -> Set[int]:
    """Evaluate a postfix Boolean expression using posting sets.

    This provided helper is adapted from the Week 04 lecture demo. It expects
    postfix input and does not perform the infix-to-postfix conversion.
    """
    stack: List[Set[int]] = []

    # The AND/OR cases are provided from the lecture demo. Complete unary NOT.
    for token in postfix_tokens:
        op = token

        if op in {"AND", "OR"}:
            right = stack.pop()
            left = stack.pop()
            if op == "AND":
                stack.append(left & right)
            else:
                stack.append(left | right)
        elif op == "NOT":
            # (Task 1): implement unary NOT.
            operand = stack.pop()
            stack.append(set(get_all_doc_ids(index_path)) - operand)
        else:
            stack.append(set(get_posting_list(token, index_path)))

    return stack.pop()


def process_boolean_query(query: str, index_path: str) -> Set[int]:
    """Evaluate a well-formed Boolean query.

    Args:
        query: Boolean query string using unigram operands, uppercase operators
            (``AND``, ``OR``, ``NOT``), and optional parentheses.
        index_path: Path to the unified index package created by Task 1.

    Returns:
        Set of document IDs matching the query.

    Semantics:
        ``NOT`` has highest precedence, then ``AND``, then ``OR``. Parentheses
        override precedence. Query operands are unigrams.
    """
    # Boolean overall process:
    # infix query -> tokens -> postfix expression -> stack evaluation
    # -> document IDs
    #
    # See the Week 04 lecture sections "Shunting-Yard Algorithm" and
    # "Postfix Evaluation". The postfix evaluator above is provided from the
    # lecture demo; complete the infix-to-postfix stage and connect the stages.
    #
    # If Task 1 temporarily blocks progress, continue with Task 2 and the
    # Task 3 baseline, then return here before final submission.
    tokens = query.split()
    postfix_tokens = _shunting_yard_reorder(tokens)
    return _evaluate_postfix(postfix_tokens, index_path)
