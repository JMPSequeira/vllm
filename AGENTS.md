# Agent Instructions for vLLM

> These instructions apply to **all** AI-assisted contributions to `vllm-project/vllm`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
    - Why this is not duplicating an existing PR.
    - Test commands run and results.
    - Model evaluation results when the change affects output, accuracy, or serving.
    - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

- **Never use system `python3` or bare `pip`/`pip install`.** All Python commands must go through `uv` and `.venv/bin/python`.

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

# Always make sure `pre-commit` and its hooks are installed:
uv pip install -r requirements/lint.txt
pre-commit install
```

### Installing dependencies

```bash
# Start with precompiled artifacts for an editable install:
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

For C/C++ or CUDA changes, follow the
[incremental compilation workflow](docs/contributing/incremental_build.md) to
configure and perform incremental builds.

### Tests

> Requires [Environment setup](#environment-setup) and [Installing dependencies](#installing-dependencies).

```bash
# Install test dependencies (use cuda.in on non-x86_64):
uv pip install -r requirements/test/cuda.in

# Run a specific test file:
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

When adding tests:

- **Design before you write.** Answer four questions first: what is the module
  for, what is its I/O contract, what failure am I guarding against, and what is
  the cheapest level that catches it (unit over integration over e2e)?
- **Reuse before create.** Extend existing test files, `conftest.py` fixtures, and
  helpers; add a new file only when no nearby suite fits.
- **Test behavior with intent.** Assert observable outcomes through public APIs;
  state why in the name or docstring. Skip trivial wiring; flaky tests are worse
  than no tests.
- **Keep it minimal.** One behavior per test and the smallest setup that
  triggers it; if the test diff dwarfs the code change, cut scope.
- **No one-off kernel benchmarks in `tests/`.** Put kernel perf work in
  `benchmarks/kernels/`; prove correctness in existing pytest suites.
- **Run model evals for model-affecting changes.** Search `tests/evals/` or use
  `vllm bench` and include results in the PR — do not wait for reviewers to ask.

For model-specific requirements, see
[`docs/contributing/model/tests.md`](docs/contributing/model/tests.md).

### Running linters

> Requires [Environment setup](#environment-setup).

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff-check --all-files

# Run mypy as it is in CI:
pre-commit run mypy-3.12 --all-files --hook-stage manual
```

The line length limit for Python code is 88 characters. If you are not sure, use pre-commit to check.

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) (`Args:`/`Returns:`/`Raises:` sections), not reStructuredText/Sphinx fields (`:param:`, `:return:`, `:rtype:`).

### Coding style guidelines

- Match existing code style
- Minimize use of comments. Eliminate comments which are redundant, preferring legible and self-documenting code. When used, keep docstrings and comments brief and direct.
- Assume the reader is familiar with vLLM.

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`):

```text
Your commit message here

Co-authored-by: Agent Name Here
Signed-off-by: Your Name <your.email@example.com>
```

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

Security reviewers should start with [`SECURITY.md`](SECURITY.md),
[`docs/usage/security.md`](docs/usage/security.md), and
[`docs/contributing/vulnerability_management.md`](docs/contributing/vulnerability_management.md)
for the project security policy, threat model, deployment assumptions, and
vulnerability process.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.

## LIL integration container changelog

Before pushing or importing a runtime-affecting change into an
`integration/*` branch, read [`.lil/README.md`](.lil/README.md) and add the
required `.lil/changes/<id>.json` fragment to the same commit series. Preserve
published fragments unchanged; corrections use a new fragment.

## Write Without Hidden Context

All repository prose—documentation, comments, docstrings, commit and PR text,
plans, reports, names, errors, and technical summaries—must make sense to a
technically capable reader who has the repository but none of the conversation
or development history.

- Describe the system as it exists. State its purpose, behavior, invariants,
  interfaces, evidence, and limitations directly. Do not narrate the journey.
- Do not use lifecycle labels as identities: `Phase 2`, `pilot`, `next`,
  `current`, `new`, `old`, `latest`, and similar terms are not technical names.
- Do not use false definite references. Phrases such as `the capture`,
  `the experiment`, or `this approach` are invalid unless the exact object was
  introduced locally and unambiguously. Counts, dates, and versions are
  attributes, not identities.
- On first reference, give the object's semantic role and, when relevant, its
  durable identifier: artifact name, path, schema, revision, manifest, or hash.
- Explain concepts before identifiers. Do not make internal codenames,
  experiment labels, profile numbers, or implementation shorthand the
  vocabulary of the design. Mention literal identifiers only after describing
  what they mean and only when the reader must use them.
- Canonical documentation is a present-state specification, not a changelog.
  Replace stale claims instead of layering history on top. Put chronology,
  rejected attempts, and retrospectives only in explicitly historical
  documents.
- Do not include meta-commentary about why a statement is present, what kind of
  claim it is, how it relates to the prompt, or what the author is or is not
  asserting. State the substantive fact, requirement, assumption, or limitation
  directly. If a distinction matters technically, encode it in the specification
  or evidence itself rather than commenting on the prose.
- State evidence as conditions, measurement, result, and conclusion—not as a story.
- Comments explain invariants, intent, and non-obvious constraints, never change
  history.
- Commits and PRs state the resulting behavior, technical reason, compatibility
  impact, and validation. They do not recount attempts or pivots.
- A document should contain conclusions and technical content, never traces of the
  author managing the reader's interpretation of that content.

Final test: if understanding any sentence requires "you had to be there,"
rewrite it.
