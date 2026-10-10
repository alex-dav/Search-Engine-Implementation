#!/usr/bin/env python3
# test_sanity/check_submission.py
# Minimal public sanity checks:
# - Ensures required modules/functions import
# - Ensures basic runnability (no crashes)
# - Ensures a SMALL set of public semantics on a tiny corpus (necessary, not sufficient)
#
# Note: Passing this script does NOT guarantee full marks. The grading tests include more cases.

import ast
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Any, Optional

from importlib import metadata as importlib_metadata


REPO_ROOT = Path(__file__).resolve().parent.parent
# Keep generated checks and NLTK data out of the submitted repository. Python
# selects a writable per-user system temp directory on Windows, macOS, and
# Linux. The repository fallback preserves runnability on restricted setups.
try:
    TMP_ROOT = Path(tempfile.gettempdir()) / "ir_ass1_partB_sanity"
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    TMP_DIR = Path(tempfile.mkdtemp(prefix="run_", dir=TMP_ROOT))
    NLTK_DATA_DIR = TMP_ROOT / "nltk_data"
    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    TMP_ROOT = REPO_ROOT / "test_sanity" / "_tmp"
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    TMP_DIR = Path(tempfile.mkdtemp(prefix="run_", dir=TMP_ROOT))
    NLTK_DATA_DIR = TMP_ROOT / "nltk_data"
    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
atexit.register(shutil.rmtree, TMP_DIR, ignore_errors=True)
os.environ["NLTK_DATA"] = str(NLTK_DATA_DIR)

# ---- tiny demo corpus for Task 1/2/3 sanity (fast) ----
DOC_IDS: List[int] = [10, 20, 30]
TOKENIZED_DOCS: List[List[str]] = [
    ["climate", "change"],       # doc 10
    ["machine", "learning"],     # doc 20
    ["climate", "policy"],       # doc 30
]

# Task 1 query-processing sanity cases (tiny corpus)
Q_BOOL_1 = "climate AND change"                 # -> {10}
Q_BOOL_2 = "climate OR machine"                 # -> {10,20,30}
Q_BOOL_3 = "climate AND NOT learning"           # -> {10,30}
Q_WC_1 = "cl*"                                   # -> {10,30}
Q_WC_2 = "*ng"                                   # -> {20}
Q_WC_3 = "learn*ing"                             # -> {20}

RESULTS: List[Tuple[str, str, str]] = []  # (name, status, msg)

INDEX_PATH = TMP_DIR / "index_pkg.pkl"

# ============================================================
# Runtime policy (spec)
# ============================================================
MAX_WALLCLOCK_SECONDS = 300.0  # 5-minute wall-clock limit
IMPORT_HELP_URL = (
    "https://rmit.instructure.com/courses/157398/pages/"
    "python-project-imports-and-running-modules"
)
LOCAL_PROJECT_PACKAGES = {
    "index",
    "metrics",
    "query_processing",
    "ranking",
    "system",
    "test_sanity",
    "utils",
}


def record(name: str, status: str, msg: str = ""):
    # status in {"PASS","FAIL","WARN","SKIP"}
    RESULTS.append((name, status, msg))
    print(f"[{status}] {name}" + (f" - {msg}" if msg else ""))


def _import_failure_guidance(error: Exception) -> str:
    """Return focused help for common project-import failures."""
    message = str(error)

    if (
        "attempted relative import with no known parent package" in message
        or "beyond top-level package" in message
    ):
        advice = (
            "Run the assignment's python -m command from the starter root "
            "instead of running an individual task file."
        )
    elif isinstance(error, ModuleNotFoundError):
        missing = str(error.name or "").split(".", maxsplit=1)[0]
        if missing in LOCAL_PROJECT_PACKAGES:
            advice = (
                f"Python could not find the local '{missing}' package. Open "
                "the extracted starter folder itself and run from that root; "
                "do not add a local path to sys.path."
            )
        else:
            advice = (
                f"Python could not find '{error.name}'. Check the import "
                "spelling and confirm that the selected .venv contains any "
                "required third-party package."
            )
    elif isinstance(error, ImportError) and "cannot import name" in message:
        advice = (
            "Python found the module, but not the requested name. Check the "
            "function name and the file where it is defined."
        )
    else:
        advice = "Check the original error raised while importing this module."

    return f"{advice} Import help: {IMPORT_HELP_URL}"


def import_or_fail(module_path: str):
    try:
        return __import__(module_path, fromlist=["*"])
    except Exception as e:
        guidance = _import_failure_guidance(e)
        raise ImportError(
            f"Import failed for '{module_path}': {e}\n{guidance}"
        ) from e


def _is_sorted_unique_int_list(xs: Any) -> bool:
    return isinstance(xs, list) and all(isinstance(x, int) for x in xs) and xs == sorted(set(xs))


def _is_sorted_unique_str_list(xs: Any) -> bool:
    return isinstance(xs, list) and all(isinstance(x, str) for x in xs) and xs == sorted(set(xs))


def _fail_if_not(condition: bool, msg: str):
    if not condition:
        raise AssertionError(msg)


def _failure_detail(error: Exception) -> str:
    """Give unfinished starter functions a useful public failure message."""
    message = str(error).strip()
    if isinstance(error, NotImplementedError) and not message:
        return "Not implemented yet"
    return message


def _get_dist_version(dist_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _get_pkg_version(import_name: str) -> Optional[str]:
    # import name -> probable distribution names
    dist_candidates = {
        "numpy": ["numpy"],
        "nltk": ["nltk"],
        "bs4": ["beautifulsoup4", "bs4"],
        "sklearn": ["scikit-learn", "sklearn"],
        "pandas": ["pandas"],
        "scipy": ["scipy"],
    }.get(import_name, [import_name])

    for dist in dist_candidates:
        v = _get_dist_version(dist)
        if v is not None:
            return v

    # fallback: module.__version__
    try:
        mod = __import__(import_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def _check_wallclock_or_fail(start_t: float, step_label: str):
    """Fail fast if total runtime exceeds the wall-clock limit."""
    elapsed = time.perf_counter() - start_t
    if elapsed > MAX_WALLCLOCK_SECONDS:
        record(
            "Runtime policy: wall-clock limit",
            "FAIL",
            f"Exceeded {MAX_WALLCLOCK_SECONDS:.0f}s during '{step_label}' (elapsed={elapsed:.2f}s). "
            "The assignment applies a 5-minute wall-clock limit; your submission may be terminated if it exceeds this limit."
        )
        _print_summary_and_exit(1, title="Summary (stopped due to wall-clock limit)")


# ============================================================
# Environment gate (Python + required packages)
# ============================================================
SUPPORTED_PY_MAJOR = 3
SUPPORTED_PY_MIN_MINOR = 10
SUPPORTED_PY_MAX_MINOR = 14
REQUIRED_NLTK_VERSION = "3.9.2"

REQUIRED_IMPORTS = ["numpy", "nltk", "bs4"]
PERMITTED_THIRD_PARTY_IMPORTS = {"numpy", "nltk", "bs4"}
IMPORT_SCAN_IGNORED_DIRS = {
    ".git", ".venv", ".conda", "venv", "env", "__pycache__",
    "cache", "data", "runs", "evaluation_results", "_tmp",
}


def step_environment_gate():
    """
    Hard gate: fail fast if Python is unsupported or required packages are missing.
    This avoids "works on my machine" submissions.
    """
    name = "Environment: Python must be 3.10 to 3.14"
    vi = sys.version_info
    try:
        supported = (
            vi.major == SUPPORTED_PY_MAJOR
            and SUPPORTED_PY_MIN_MINOR <= vi.minor <= SUPPORTED_PY_MAX_MINOR
        )
        _fail_if_not(
            supported,
            f"Detected Python {vi.major}.{vi.minor}.{vi.micro}. "
            f"Supported versions are Python {SUPPORTED_PY_MAJOR}.{SUPPORTED_PY_MIN_MINOR} "
            f"to {SUPPORTED_PY_MAJOR}.{SUPPORTED_PY_MAX_MINOR} inclusive.\n"
            "Fix: select a supported Python environment and re-run."
        )
        record(name, "PASS", f"Python {vi.major}.{vi.minor}.{vi.micro}")
    except Exception as e:
        record(name, "FAIL", str(e))
        raise

    name = "Environment: required packages import (numpy, nltk, bs4)"
    try:
        for pkg in REQUIRED_IMPORTS:
            __import__(pkg)

        numpy_v = _get_pkg_version("numpy")
        nltk_v = _get_pkg_version("nltk")
        bs4_v = _get_pkg_version("bs4")
        _fail_if_not(
            nltk_v == REQUIRED_NLTK_VERSION,
            f"Detected NLTK {nltk_v or 'unknown'}. Required NLTK {REQUIRED_NLTK_VERSION}.\n"
            "Fix: run `python -m pip install -r requirements.txt`."
        )
        record(name, "PASS", f"numpy={numpy_v}; nltk={nltk_v}; beautifulsoup4={bs4_v}")
    except Exception as e:
        record(
            name,
            "FAIL",
            f"Missing/broken required package import. Details: {e}\n"
            "Fix: run `python -m pip install -r requirements.txt` in a supported Python environment."
        )
        raise


def step_nltk_resources():
    """Download the small NLTK resources used by common preprocessing choices."""
    name = "Environment: NLTK data (punkt, punkt_tab, stopwords)"
    resources = (
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    )
    try:
        import nltk

        downloaded = []
        for resource_path, package_name in resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                ok = nltk.download(
                    package_name,
                    download_dir=str(NLTK_DATA_DIR),
                    quiet=True,
                )
                _fail_if_not(ok, f"Could not download NLTK resource: {package_name}")
                nltk.data.find(resource_path)
                downloaded.append(package_name)

        detail = (
            "Downloaded: " + ", ".join(downloaded)
            if downloaded
            else "All resources already available"
        )
        record(name, "PASS", detail)
    except Exception as e:
        record(name, "FAIL", f"Could not prepare NLTK data. Details: {e}")
        raise


def step_permitted_imports_scan():
    """Check ordinary imports in submitted Python files against the spec."""
    name = "Submission: Python imports use permitted libraries"
    local_roots = set(LOCAL_PROJECT_PACKAGES)
    for child in REPO_ROOT.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_file() and child.suffix == ".py":
            local_roots.add(child.stem)
        elif child.is_dir():
            local_roots.add(child.name)

    permitted = (
        set(sys.stdlib_module_names)
        | set(sys.builtin_module_names)
        | PERMITTED_THIRD_PARTY_IMPORTS
        | local_roots
        | {"__future__"}
    )
    non_permitted = {}
    syntax_errors = []

    for path in sorted(REPO_ROOT.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT)
        if any(
            part in IMPORT_SCAN_IGNORED_DIRS or part.startswith(".")
            for part in relative.parts[:-1]
        ):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
        except (OSError, UnicodeError, SyntaxError) as error:
            syntax_errors.append(f"{relative}: {error}")
            continue

        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported_roots.add(node.module.split(".", 1)[0])

        for imported_root in sorted(imported_roots - permitted):
            non_permitted.setdefault(imported_root, []).append(str(relative))

    problems = []
    if non_permitted:
        details = [
            f"{module} in {', '.join(paths)}"
            for module, paths in sorted(non_permitted.items())
        ]
        problems.append("Non-permitted imports: " + "; ".join(details))
    if syntax_errors:
        problems.append("Could not inspect imports: " + "; ".join(syntax_errors))

    if problems:
        record(
            name,
            "FAIL",
            " ".join(problems) + ". Permitted imports are numpy, nltk, bs4, "
            "the Python standard library, and local project modules."
        )
    else:
        record(name, "PASS", "No non-permitted imports found")


def step_task1_build_index():
    name = "Task1: create_all_indexes() builds one package"
    try:
        builders = import_or_fail("index.builders")
        _fail_if_not(hasattr(builders, "create_all_indexes"),
                     "Missing function: index.builders.create_all_indexes(...)")

        # Fresh build
        if INDEX_PATH.exists():
            _fail_if_not(INDEX_PATH.is_file(), f"Index path exists but is not a file: {INDEX_PATH}")
            INDEX_PATH.unlink()

        builders.create_all_indexes(TOKENIZED_DOCS, str(INDEX_PATH), doc_ids=DOC_IDS)
        try:
            access = import_or_fail("index.access")
            if hasattr(access, "clear_package_cache"):
                access.clear_package_cache(str(INDEX_PATH))
        except Exception:
            pass
        _fail_if_not(INDEX_PATH.exists(), f"Index package not created at {INDEX_PATH}")
        _fail_if_not(INDEX_PATH.is_file(), f"Index path exists but is not a file: {INDEX_PATH}")
        record(name, "PASS")
    except Exception as e:
        record(name, "FAIL", str(e))


def step_task1_access():
    try:
        access = import_or_fail("index.access")
    except Exception as e:
        record("Task1: import index.access", "FAIL", str(e))
        return

    # ---- get_posting_list ----
    name = "Task1: get_posting_list('climate') returns sorted+dedup correct docs"
    try:
        _fail_if_not(hasattr(access, "get_posting_list"),
                     "Missing function: index.access.get_posting_list(...)")
        pl = access.get_posting_list("climate", str(INDEX_PATH))
        _fail_if_not(_is_sorted_unique_int_list(pl),
                     f"Expected sorted+dedup List[int], got: {pl!r}")
        expected = [10, 30]
        _fail_if_not(pl == expected, f"Expected {expected}, got {pl}")
        record(name, "PASS")
    except Exception as e:
        record(name, "FAIL", str(e))

    name = "Task1: get_posting_list(OOV) returns []"
    try:
        # Use a term not present in the tiny corpus
        pl = access.get_posting_list("algorithm", str(INDEX_PATH))
        _fail_if_not(isinstance(pl, list), f"Expected List[int], got: {type(pl)}")
        _fail_if_not(pl == [], f"Expected [], got {pl}")
        record(name, "PASS")
    except Exception as e:
        record(name, "FAIL", str(e))

    # ---- find_wildcard_matches ----
    name = r"Task1: find_wildcard_matches('$cl') returns sorted+dedup terms"
    try:
        _fail_if_not(hasattr(access, "find_wildcard_matches"),
                     "Missing function: index.access.find_wildcard_matches(...)")
        matches = access.find_wildcard_matches("$cl", str(INDEX_PATH))
        _fail_if_not(_is_sorted_unique_str_list(matches),
                     f"Expected sorted+dedup List[str], got: {matches!r}")
        _fail_if_not(matches == ["climate"], f"Expected ['climate'], got {matches}")
        record(name, "PASS")
    except Exception as e:
        record(name, "FAIL", str(e))

def step_task1_processors() -> dict:
    name_base = "Task1"
    readiness = {"boolean": False, "wildcard": False}
    try:
        bo = import_or_fail("query_processing.boolean")
        wc = import_or_fail("query_processing.wildcard")
    except Exception as e:
        record(f"{name_base}: import boolean/wildcard modules", "FAIL", str(e))
        return readiness

    # Boolean required
    def _check_bool(q: str, expected: set, label: str):
        name = f"{name_base}: process_boolean_query {label}"
        try:
            _fail_if_not(hasattr(bo, "process_boolean_query"),
                         "Missing function: query_processing.boolean.process_boolean_query")
            res = bo.process_boolean_query(q, str(INDEX_PATH))
            _fail_if_not(isinstance(res, set) and all(isinstance(x, int) for x in res),
                         f"Expected Set[int], got: {res!r}")
            _fail_if_not(res == expected, f"Expected {expected}, got {res}")
            record(name, "PASS")
            return True
        except Exception as e:
            record(name, "FAIL", _failure_detail(e))
            return False

    readiness["boolean"] = all([
        _check_bool(Q_BOOL_1, {10}, f"('{Q_BOOL_1}')"),
        _check_bool(Q_BOOL_2, {10, 20, 30}, f"('{Q_BOOL_2}')"),
        _check_bool(Q_BOOL_3, {10, 30}, f"('{Q_BOOL_3}')"),
    ])

    # Wildcard required
    def _check_wc(q: str, expected: set, label: str):
        name = f"{name_base}: process_wildcard_query {label}"
        try:
            _fail_if_not(hasattr(wc, "process_wildcard_query"),
                         "Missing function: query_processing.wildcard.process_wildcard_query")
            res = wc.process_wildcard_query(q, str(INDEX_PATH))
            _fail_if_not(isinstance(res, set) and all(isinstance(x, int) for x in res),
                         f"Expected Set[int], got: {res!r}")
            _fail_if_not(res == expected, f"Expected {expected}, got {res}")
            record(name, "PASS")
            return True
        except Exception as e:
            record(name, "FAIL", _failure_detail(e))
            return False

    readiness["wildcard"] = all([
        _check_wc(Q_WC_1, {10, 30}, f"('{Q_WC_1}')"),
        _check_wc(Q_WC_2, {20}, f"('{Q_WC_2}')"),
        _check_wc(Q_WC_3, {20}, f"('{Q_WC_3}')"),
    ])
    return readiness

def step_task2_ranker():
    name = "Task2: rank_documents(...) returns full permutation + aligned scores"
    try:
        rankers = import_or_fail("ranking.rankers")
        _fail_if_not(hasattr(rankers, "rank_documents"),
                     "Missing function: ranking.rankers.rank_documents(...)")

        query_toks = ["climate", "change"]
        candidate_docs = TOKENIZED_DOCS[:]  # aligned with DOC_IDS
        ranked_ids1, scores1 = rankers.rank_documents(
            query_toks, candidate_docs, DOC_IDS, str(INDEX_PATH), method="default"
        )
        ranked_ids2, scores2 = rankers.rank_documents(
            query_toks, candidate_docs, DOC_IDS, str(INDEX_PATH), method="default"
        )

        _fail_if_not(isinstance(ranked_ids1, list) and isinstance(scores1, list),
                     "rank_documents should return Tuple[List[int], List[float]]")
        _fail_if_not(len(ranked_ids1) == len(scores1),
                     "ranked_doc_ids and scores must have same length")
        _fail_if_not(sorted(ranked_ids1) == sorted(DOC_IDS),
                     "ranked_doc_ids must be a permutation of input doc_ids")
        _fail_if_not(all(isinstance(x, int) for x in ranked_ids1),
                     "ranked_doc_ids must be List[int]")
        _fail_if_not(all(isinstance(x, (int, float)) for x in scores1),
                     "scores must be numeric")
        ranked_pairs = list(zip(ranked_ids1, scores1))
        expected_pairs = sorted(
            ranked_pairs,
            key=lambda item: (-float(item[1]), int(item[0])),
        )
        _fail_if_not(
            ranked_pairs == expected_pairs,
            "ranked_doc_ids and scores must be ordered by descending score, "
            "with ties broken by ascending doc_id",
        )
        _fail_if_not(
            ranked_ids1 == ranked_ids2 and scores1 == scores2,
            "Repeated calls with the same input must return identical results",
        )

        record(name, "PASS", f"ranked {len(ranked_ids1)} docs")
    except Exception as e:
        record(name, "FAIL", str(e))


def _check_task3_query_preservation(task1_ready: dict, d_path: Path) -> None:
    """Check that the baseline preserves structured query syntax."""
    cases = [
        ("boolean", "QB", "climate AND NOT policy", {10}),
        ("wildcard", "QW", "clim*", {10, 30}),
    ]
    active_cases = []
    for category, qid, query, expected in cases:
        name = f"Task3: no_optimisation preserves {category} query syntax"
        if not task1_ready.get(category, False):
            record(
                name,
                "SKIP",
                "Complete the matching Task 1 processor first. This check only "
                "tests whether the baseline pipeline preserves a processor that "
                "already works.",
            )
            continue
        active_cases.append((category, qid, query, expected))

    if not active_cases:
        return

    q_path = TMP_DIR / "queries_query_preservation.json"
    out_path = TMP_DIR / "run_query_preservation_no_optimisation.json"
    queries = [
        {"qid": qid, "query": query}
        for _, qid, query, _ in active_cases
    ]
    q_path.write_text(json.dumps(queries, indent=2), encoding="utf-8")
    if out_path.exists():
        out_path.unlink()

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "system.search_system",
        str(q_path),
        str(d_path),
        str(out_path),
        "no_optimisation",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=90,
    )
    if proc.returncode != 0 or not out_path.exists():
        detail = (
            f"The no_optimisation CLI could not complete this integration check. "
            f"STDOUT: {proc.stdout or '(empty)'} STDERR: {proc.stderr or '(empty)'}"
        )
        for category, _, _, _ in active_cases:
            record(
                f"Task3: no_optimisation preserves {category} query syntax",
                "FAIL",
                detail,
            )
        return

    data = json.loads(out_path.read_text(encoding="utf-8"))
    by_qid = {
        str(item.get("qid")): item
        for item in data
        if isinstance(item, dict)
    }
    failure_message = (
        "Task 3 baseline query-preservation check failed. The Task 1 processor "
        "passes separately, but no_optimisation changes the query before query "
        "processing. Preserve Boolean operators such as AND, OR, and NOT, "
        "parentheses such as ( climate OR policy ), and wildcard syntax such "
        "as learn*ing when preparing the query."
    )
    for category, qid, _, expected in active_cases:
        name = f"Task3: no_optimisation preserves {category} query syntax"
        item = by_qid.get(qid, {})
        actual = item.get("doc_ids")
        if isinstance(actual, list) and set(actual) == expected:
            record(name, "PASS")
        else:
            record(name, "FAIL", failure_message)


def step_task3_cli(task1_ready: dict):
    try:
        # Mini input for fast sanity (avoid full dev runtime)
        q_path = TMP_DIR / "queries_sanity.json"
        d_path = TMP_DIR / "documents_sanity.jsonl"
        queries = [
            {"qid": "Q1", "query": "climate change"},
            {"qid": "Q2", "query": "machine learning"},
        ]
        docs = [
            {"id": 10, "text": "climate change"},
            {"id": 20, "text": "machine learning"},
            {"id": 30, "text": "climate policy"},
        ]
        q_path.write_text(json.dumps(queries, indent=2), encoding="utf-8")
        d_path.write_text("\n".join(json.dumps(d) for d in docs) + "\n", encoding="utf-8")
    except Exception as e:
        record("Task3: prepare CLI sanity inputs", "FAIL", str(e))
        return

    for method in ("no_optimisation", "default"):
        name = f"Task3: CLI produces valid {method} run JSON"
        try:
            out_path = TMP_DIR / f"run_sanity_{method}.json"
            if out_path.exists():
                out_path.unlink()

            cmd = [sys.executable, "-u", "-m", "system.search_system",
                   str(q_path), str(d_path), str(out_path), method]

            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=90,
            )
            if proc.returncode != 0:
                usage_hint = ""
                stdout = proc.stdout or ""
                stderr = proc.stderr or ""
                if "Usage:" in stdout or "Usage:" in stderr:
                    usage_hint = (
                        "Hint: CLI is invoked as `python -m system.search_system <queries_json> "
                        "<documents_jsonl> <run_output_json> <method>` (4 args; method passed as 4th).\n"
                        "Ensure your CLI accepts Task 3 methods `no_optimisation` and `default`."
                    )
                if method == "no_optimisation":
                    usage_hint += (
                        "\nThe `no_optimisation` method is the baseline system used for comparison; "
                        "keep it runnable while developing your `default` final system."
                    )
                raise RuntimeError(
                    f"CLI method={method!r} exited {proc.returncode}\n"
                    f"STDOUT:\n{stdout or '(empty)'}\n"
                    f"STDERR:\n{stderr or '(empty)'}\n"
                    f"{usage_hint}"
                )

            if not out_path.exists():
                raise FileNotFoundError(f"CLI did not produce the output JSON for method={method!r}")

            data = json.loads(out_path.read_text(encoding="utf-8"))
            _fail_if_not(isinstance(data, list) and len(data) == len(queries),
                         f"Output JSON for method={method!r} must be a list of {len(queries)} result objects, got {type(data)} len={len(data) if isinstance(data,list) else 'n/a'}")

            allowed_doc_ids = {d["id"] for d in docs}
            expected_qids = {q["qid"] for q in queries}
            seen_qids = set()

            for item in data:
                _fail_if_not(isinstance(item, dict), f"Each result must be an object, got {type(item)}")
                _fail_if_not("qid" in item and "doc_ids" in item, "Each result must include 'qid' and 'doc_ids'")
                _fail_if_not(isinstance(item["qid"], str), "qid must be str")
                _fail_if_not(item["qid"] not in seen_qids, f"Duplicate qid in output: {item['qid']}")
                seen_qids.add(item["qid"])

                doc_ids = item["doc_ids"]
                _fail_if_not(isinstance(doc_ids, list) and all(isinstance(x, int) for x in doc_ids),
                             "doc_ids must be List[int]")
                _fail_if_not(len(doc_ids) <= 10, "doc_ids must be top-10 (len <= 10)")
                _fail_if_not(len(doc_ids) == len(set(doc_ids)), "doc_ids must not contain duplicates")
                _fail_if_not(set(doc_ids).issubset(allowed_doc_ids),
                             f"doc_ids contain unknown ids: {set(doc_ids) - allowed_doc_ids}")

                if "scores" in item:
                    scores = item["scores"]
                    _fail_if_not(isinstance(scores, list) and len(scores) == len(doc_ids),
                                 "scores must be a list aligned 1:1 with doc_ids")
                    _fail_if_not(all(isinstance(s, (int, float)) for s in scores),
                                 "scores must be numeric")

            _fail_if_not(
                seen_qids == expected_qids,
                f"Output qids must exactly match the input queries. Missing={sorted(expected_qids - seen_qids)}, extra={sorted(seen_qids - expected_qids)}",
            )

            record(name, "PASS", f"{method} produced valid run JSON")
        except Exception as e:
            record(name, "FAIL", str(e))

    _check_task3_query_preservation(task1_ready, d_path)


def _print_summary_and_exit(exit_code: int, title: str = "Summary"):
    print(f"\n=== {title} ===")
    failed = 0
    for name, status, msg in RESULTS:
        print(f"{status:4} - {name}")
        if msg:
            print(f"       - {msg}")
        if status == "FAIL":
            failed += 1

    passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
    warned = sum(1 for _, s, _ in RESULTS if s == "WARN")
    skipped = sum(1 for _, s, _ in RESULTS if s == "SKIP")
    print(f"\nTotal: {passed} passed, {warned} warned, {skipped} skipped, {failed} failed")
    print("Note: Task 2 sanity checks verify the interface, not ranking quality.")
    print("Run `python -m metrics.eval_pear default` to view your dev Pearson score.")
    sys.exit(exit_code)


def main():
    print("=== Sanity Check: starting ===")
    start_t = time.perf_counter()

    # 0) Environment hard gate (fail fast)
    try:
        step_environment_gate()
        _check_wallclock_or_fail(start_t, "Environment gate")
        step_nltk_resources()
        _check_wallclock_or_fail(start_t, "NLTK data setup")
        step_permitted_imports_scan()
        _check_wallclock_or_fail(start_t, "Permitted import scan")
    except Exception:
        _print_summary_and_exit(1, title="Summary (stopped due to environment failure)")

    # After env is verified, allow local imports from repo root
    sys.path.insert(0, str(REPO_ROOT))

    step_task1_build_index()
    _check_wallclock_or_fail(start_t, "Task1 build index")

    step_task1_access()
    _check_wallclock_or_fail(start_t, "Task1 access functions")

    task1_ready = step_task1_processors()
    _check_wallclock_or_fail(start_t, "Task1 processors")

    step_task2_ranker()
    _check_wallclock_or_fail(start_t, "Task2 ranker")

    step_task3_cli(task1_ready)
    _check_wallclock_or_fail(start_t, "Task3 CLI")

    elapsed = time.perf_counter() - start_t
    record("Runtime: total wall-clock", "PASS", f"{elapsed:.2f}s (limit {MAX_WALLCLOCK_SECONDS:.0f}s)")

    # Final summary + exit code
    failed = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    _print_summary_and_exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
