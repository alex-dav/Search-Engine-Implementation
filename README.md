# ISYS1079 (Undergraduate) - Assignment 1 Part B

This is the Search Engine Implementation starter for Managing Semi-structured and Unstructured Data. It follows the required module layout and command-line interfaces used by the assignment commands and sanity checker.

Part B builds on Part A. Your submission must meaningfully reuse or adapt your own Part A work in at least one relevant implementation. Simply retaining unused utility files does not meet this requirement. You may copy the completed Part A utility files you use into this repository because the shared function contracts are kept consistent.
Part B includes a system dev collection under `data/dev/` and a dedicated Task 2 ranking collection under `data/dev\_ranking/`. Task 2 uses fixed tokenization so its Pearson score measures ranking rather than Part A preprocessing. You may use Part A preprocessing as part of a Task 3 system optimisation.

## Setup

The VS Code steps below provide a guided setup. If you are comfortable with Python environments or the terminal, use your preferred setup and skip to the quick checks. In either case, use Python 3.10 to 3.14, install `requirements.txt` and the required NLTK resources, and follow the permitted-import list in the specification.

For additional help, see [Python Setup and Troubleshooting](https://rmit.instructure.com/courses/157398/pages/python-setup-and-troubleshooting?module_item_id=8766153).

For guided setup in VS Code:

1. Download the starter ZIP from Canvas.
2. Extract the ZIP first. On Windows, right-click it and select **Extract All**. On macOS, double-click it. Do not work inside the ZIP.
3. Open VS Code, select **File > Open Folder**, and choose the extracted `ir-ass1-partB-startcode-UG` folder itself.
4. If VS Code displays **Restricted Mode**, select **Manage** and trust the extracted course folder. Confirm that `README.md`, `requirements.txt`, `index/`, and `query\_processing/` are visible in the VS Code file list.
5. Press **Ctrl+Shift+P** on Windows or **Cmd+Shift+P** on macOS. Type `Python: Create Environment` and select it from the results. On the next screen, do not select **Quick Create**. Select **venv**, the second option. If the command does not appear, install or update the Microsoft Python and Python Environments extensions in VS Code.
6. Select an installed Python version from 3.10 to 3.14. If no supported version is listed, install one from [python.org](https://www.python.org/downloads/) and reopen VS Code.
7. Select **Install project dependencies**. On the next screen, keep `requirements.txt` selected and click **OK**. VS Code will create a local `.venv`, install the packages, and select the new environment for this folder.
8. When environment creation finishes, select **Terminal > New Terminal**. The terminal prompt should begin with `(.venv)`, which confirms that the new environment is active. Then run:

```bash
python -m nltk.downloader punkt punkt\_tab stopwords
```

The self-checker also attempts to download any missing NLTK tokenizer or stopword data.

On Windows, if PowerShell reports that script execution is disabled, select **Terminal > Select Default Profile > Command Prompt**, then open a new terminal. You do not need to change the system execution policy.

For help importing functions between project folders or resolving module errors, see [Python Project Imports and Running Modules](https://rmit.instructure.com/courses/157398/pages/python-project-imports-and-running-modules).

## Quick Checks

Run the following commands from the same VS Code terminal. Its current folder must be the extracted starter folder containing this `README.md`.

```bash
python -m test\_sanity.check\_submission

# In the starter, default initially points to keyword\_match.
python -m metrics.eval\_pear keyword\_match
python -m metrics.eval\_pear default

python -m system.search\_system data/dev/queries.json data/dev/documents.jsonl runs/run\_no\_optimisation.json no\_optimisation

# In the starter, default initially matches no\_optimisation.
python -m system.search\_system data/dev/queries.json data/dev/documents.jsonl runs/run\_default.json default

python -m metrics.eval\_map
```

The sanity checker verifies interfaces and small examples. It does not measure ranking quality. Use `metrics.eval\_pear` for Task 2 dev guidance and `metrics.eval\_map` for Task 3 dev guidance.

## Project Folders

```text
index/               index construction and access
query\_processing/    structured query processors
ranking/             Task 2 ranker
system/              Task 3 retrieval pipeline
metrics/             development evaluators
utils/               reusable Part A utilities
data/                provided development data
runs/                generated run files
test\_sanity/         public self-checker
```

See the Main Specification for the exact required submission layout.

