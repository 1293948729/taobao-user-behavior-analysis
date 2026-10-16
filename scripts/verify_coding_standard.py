"""补查编码规范中的文档、类型、SQL和导入关系等要求。

主要入口：main。
"""

import argparse
import ast
import re
from datetime import datetime
from pathlib import Path

from taobao_user_behavior.settings import load_runtime_settings
from taobao_user_behavior.source_inventory import local_imports, source_fingerprints
from taobao_user_behavior.storage import save_json


def _public_api(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    findings = []
    arguments = [
        argument
        for argument in [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]
        if argument.arg not in {"self", "cls"}
    ]
    if len(arguments) > 5:
        findings.append("more than five parameters")
    if (
        any(argument.annotation is None for argument in arguments)
        or node.returns is None
    ):
        findings.append("missing parameter or return annotation")
    if any(
        isinstance(value, (ast.List, ast.Dict, ast.Set))
        for value in [
            *node.args.defaults,
            *node.args.kw_defaults,
        ]
    ):
        findings.append("mutable literal default")
    if not node.name.startswith("_"):
        document = ast.get_docstring(node)
        if not document:
            findings.append("public callable lacks docstring")
        else:
            if arguments and "Args:" not in document:
                findings.append("public parameters lack Google Args")
            if (
                not (
                    isinstance(node.returns, ast.Constant)
                    and node.returns.value is None
                )
                and "Returns:" not in document
                and "Yields:" not in document
            ):
                findings.append("public return lacks Google Returns/Yields")
            if (
                any(isinstance(child, ast.Raise) for child in ast.walk(node))
                and "Raises:" not in document
            ):
                findings.append("public explicit exception lacks Google Raises")
    return findings


def _syntax_rules(node: ast.AST, path: Path) -> list[str]:
    rules = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _public_api(node)
    if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
        return _class_document(node)
    if isinstance(node, ast.Import) and len(node.names) != 1:
        rules.append("multiple modules in one import")
    elif isinstance(node, ast.ImportFrom):
        if node.level or any(alias.name == "*" for alias in node.names):
            rules.append("relative or wildcard import")
    elif isinstance(node, ast.ExceptHandler):
        if node.type is None:
            rules.append("bare except")
        if all(isinstance(statement, ast.Pass) for statement in node.body):
            rules.append("empty except")
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in {"eval", "exec"}:
            rules.append("dynamic code execution")
        if path.parent.name == "tests" and node.func.id == "print":
            rules.append("test uses print")
    return rules


def _class_document(node: ast.ClassDef) -> list[str]:
    document = ast.get_docstring(node)
    if not document:
        return ["public class lacks docstring"]
    if "Attributes:" not in document or "Examples:" not in document:
        return ["public class lacks attributes or typical usage"]
    return []


def _sql_assembly(node: ast.AST) -> bool:
    """检查SQL是否由字符串拼接而成。"""
    fragments = [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]
    sql_text = any(
        re.search(r"\b(SELECT|FROM|WHERE|CREATE|JOIN|GROUP BY)\b", value)
        for value in fragments
    )
    constructed = isinstance(node, (ast.JoinedStr, ast.BinOp)) or (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"format", "join", "replace"}
    )
    return bool(sql_text and constructed)


def _hanging_call(node: ast.AST, lines: list[str]) -> bool:
    """找到函数调用的实际左括号，检查多行参数的首行。"""
    if not isinstance(node, ast.Call):
        return False
    line_number = node.func.end_lineno
    if line_number is None or node.end_lineno is None:
        return False
    if line_number >= node.end_lineno:
        return False
    line = lines[line_number - 1].encode("utf-8")
    tail = line[node.func.end_col_offset :].decode("utf-8").lstrip()
    return (
        tail.startswith("(")
        and bool(tail[1:].strip())
        and not tail[1:].lstrip().startswith("#")
    )


def _inspect(path: Path) -> list[dict]:
    source = path.read_text(encoding="utf-8-sig")
    lines = source.splitlines()
    tree = ast.parse(source)
    findings = []
    for index, line in enumerate(lines, 1):
        if "\t" in line or len(line) > 88:
            findings.append({"line": index, "rule": "Tab or line wider than 88"})
    if not ast.get_docstring(tree):
        findings.append({"line": 1, "rule": "missing module docstring"})
    if not source.endswith("\n") or source.endswith("\n\n"):
        findings.append(
            {"line": len(source.splitlines()), "rule": "invalid final newline"}
        )
    for node in ast.walk(tree):
        if _hanging_call(node, lines):
            findings.append(
                {
                    "line": getattr(node, "lineno", 1),
                    "rule": "multiline call keeps argument on opening line",
                }
            )
        if _sql_assembly(node):
            findings.append(
                {"line": getattr(node, "lineno", 1), "rule": "SQL text assembly"}
            )
        findings.extend(
            {"line": getattr(node, "lineno", 1), "rule": rule}
            for rule in _syntax_rules(node, path)
        )
    return findings


def _import_cycles(package: Path) -> list[list[str]]:
    graph: dict[str, set[str]] = {}
    for path in package.glob("*.py"):
        graph[path.stem] = local_imports(path)
    cycles: list[list[str]] = []
    visited: set[str] = set()

    def _visit(name: str, stack: list[str]) -> None:
        if name in stack:
            cycles.append(stack[stack.index(name) :] + [name])
        elif name not in visited:
            for dependency in graph.get(name, set()):
                _visit(dependency, [*stack, name])
            visited.add(name)

    for name in graph:
        _visit(name, [])
    return cycles


def main() -> None:
    """扫描当前Python文件，保存具体问题和检查范围。

    Raises:
        SystemExit: 发现规范问题或循环导入时返回非零退出码。
    """
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    settings = load_runtime_settings(project, parser.parse_args().run_root)
    fingerprints = source_fingerprints(project)
    files = [project / name for name in fingerprints]
    findings = [
        {"file": str(path.relative_to(project)), **finding}
        for path in sorted(files)
        for finding in _inspect(path)
    ]
    cycles = _import_cycles(project / "src/taobao_user_behavior")
    report = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "passed": not findings and not cycles,
        "python_files": len(files),
        "source_sha256": fingerprints,
        "findings": findings,
        "import_cycles": cycles,
        "maximum_module_lines": max(
            len(path.read_text(encoding="utf-8").splitlines()) for path in files
        ),
        "modules_above_recommended_500": [
            str(path.relative_to(project))
            for path in files
            if len(path.read_text(encoding="utf-8").splitlines()) > 500
        ],
        "human_review_record": (
            "Review code rules alongside the supplied project standard."
        ),
        "not_proven_by_this_script": [
            "all business correctness or complete path coverage",
            "human code review and GitHub branch-protection execution",
            "arbitrary-input security or third-party model-file trust",
        ],
    }
    save_json(settings.outputs / "checks/coding_standard.json", report)
    print(f"PDF static rules passed={report['passed']}; {len(findings)} findings.")
    for finding in findings:
        print(finding)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
