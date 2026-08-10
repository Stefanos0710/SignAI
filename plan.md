# SignAI Refactoring Plan

Analysis only — no source files were touched. All findings below were verified against the actual tracked files (`git ls-files`) and, where relevant, `grep`/`diff` checks for real usage before being labeled "dead." Anything not fully verified is marked **[UNVERIFIED]** and should be double-checked before deleting.

## 0. Headline numbers (the real priority)

| Finding | Size / count | Why it matters |
|---|---|---|
| `models/` working-tree size | **8.0 GB** | `.keras` checkpoints + PNGs committed directly to git |
| `.git` object store | **6.1 GB**, almost all as **6.3 GB of loose (unpacked) objects** | Every clone/fetch of this repo ships gigabytes of binary model history |
| `logs/` (TensorBoard event files) | 44 MB across 8 model versions (v1, v15–v21, v36) | Training run artifacts committed as source |
| `SignAlphaSet/models/` | 4.2 MB + 3× `.keras` | Same pattern, smaller scale |

This dwarfs every other finding in this document. Cleaning up dead `.py` files saves kilobytes; the model/log bloat is the actual cost driver (clone time, CI minutes, disk space, GitHub repo size limits). **Section 3, Phase 0** addresses this first for a reason — everything else is secondary.

---

## 1. Dead & duplicate code

### 1.1 Confirmed dead files (no importers found anywhere in tracked `.py`/`.spec`/`.ui` files)

| File | Evidence | Action |
|---|---|---|
| `model.py` (root, 221 lines) | Old `build_seq2seq_model` architecture, last touched 2025-06-10. `grep -rn "import model"` across the repo returns nothing. Superseded by `build_seq2seq_model_multi_attention()` in `train-seq2seq.py`. The file's own body is ~45% commented-out "OLD VERSIONS OF MODELS" (v1, v2) — dead code nested inside dead code. | **Delete.** |
| `preprocessing_train_data_old.py` (root, 645 lines) | Name says it all; `preprocessing_train_data.py` (465 lines) is the live version referenced in `CLAUDE.md`/`AGENTS.md`. No importers found. | **Delete.** |
| `utils_experimental_train.py` (311 lines) | No importers found. `CLAUDE.md` calls it an "alternate pure-transformer architecture" — i.e. known-experimental, not wired into `train-seq2seq.py`. | **Confirm with user it's abandoned, then delete** (or move to a clearly-labeled `experiments/` dir if it's meant to be revisited). |
| `__pycache__/preprecessing_livedata_web.cpython-310.pyc` | A **compiled bytecode file that is tracked in git** for a source file (`preprecessing_livedata_web.py`, note the typo) that no longer exists anywhere in the tree. Last touched 2025-10-11. | **Delete the file, and delete the whole `__pycache__/` from git tracking** (see 1.4). |
| `models/doc.txt` | Freeform scratch notes ("1 4 woed test 80 vid -> 20 per wordd / bug fix / 2.wlasl 7k videos..."). Not referenced by any script. | **Delete**, or fold the 3 lines into a `CHANGELOG.md`/experiment log if the history has value. |

### 1.2 Near-duplicate build artifacts (`app/builds/`, `app/`)

| Files | Finding | Action |
|---|---|---|
| `app/SignAI - Updater.spec` vs `app/SignAI_Updater.spec` | Byte-for-byte identical except the app name (`'SignAI - Updater'` vs `'SignAI_Updater'`). `app/builds/build-updater-exe.py` hardcodes `APP_NAME = "SignAI - Updater"` (space variant) and `build-final-app.py` also references the space variant. The underscore file is newer by git date (2025-11-27 vs 2025-11-16) but **[UNVERIFIED]** whether anything still consumes it. | Ask which naming convention is current; delete the other spec. Likely delete `SignAI_Updater.spec`. |
| `app/builds/README.md` | The file is two READMEs concatenated back-to-back — a German section (lines 1–366, "Build-Optionen", "Häufige Probleme") followed by an English section (lines 367–526) covering the *same* Windows Defender / build-troubleshooting ground a second time, in a different structure, referencing a different exe name (`SignAI.exe` vs `SignAI - Desktop.exe`). Looks like a merge artifact rather than deliberate bilingual docs. | Pick one language/version, rewrite as a single coherent doc. |
| `app/builds/*.ifp` (3 files: `... build and uninstaller.ifp`, `... build installer.ifp`, `... build installer (Classic).ifp`) | Binary InstallForge project files, genuinely differ (not exact dupes), but it's unclear which is the one actually used for releases. **[UNVERIFIED]** — no script references them by name (they're likely opened manually in the InstallForge GUI). | Confirm with user which is current; delete the abandoned variant(s); move the survivor into a `packaging/` folder (see Section 2). |

### 1.3 Duplicated logic (same behavior, two independent implementations)

| Location | Finding | Action |
|---|---|---|
| Shoulder-centering/normalization in `preprocessing_train_data.py` (`center_keypoints`, `normalize_keypoints`) vs. the equivalent logic in `api/preprocessing_live_data.py` | `CLAUDE.md` explicitly documents these as producing "consistent" normalization, but they are two separately-written implementations, not a shared function. Any future tweak to the normalization math has to be made twice and can silently drift into train/inference skew — the single riskiest kind of bug in an ML pipeline. | **Do not touch casually.** This needs a careful, test-covered extraction into a shared `signai/preprocessing/normalization.py` used by both training and live inference, with a regression check (numerically diff old vs. new output on a fixed sample) before/after. Treat as its own scoped task, not part of a general cleanup pass. |
| `app/updater/updater.py` (788 lines) vs `app/updater/updater-app.py` (212 lines) | Both live under `app/updater/`; **[UNVERIFIED]** whether `updater-app.py` is a thin CLI/GUI entrypoint that calls into `updater.py`'s logic, or a parallel/older reimplementation. Worth a quick read before the roadmap's file-move phase. | Verify relationship; if `updater-app.py` duplicates logic rather than importing it, consolidate. |

### 1.4 Tracked files that should never have been committed

Everything below is currently **git-ignored going forward** but was committed *before* the ignore rule existed, so it's still sitting in history and in the working tree as tracked files:

- `__pycache__/preprecessing_livedata_web.cpython-310.pyc` (see 1.1)
- `logs/model_v*/**/events.out.tfevents.*` and `logs/model_v36/training_logs.txt` — TensorBoard run artifacts for 8 different training runs, tracked in git
- `models/checkpoint_v28_epoch_03.keras`, `models/trained_model_v{18,19,20,21,28}.keras`, `models/trained_model_v28.fixed.keras`, `models/training_history_v*.png` — exactly what `CLAUDE.md` tells future contributors *not* to do ("avoid committing new large `.keras` files — use release assets instead"), except these predate that rule
- `SignAlphaSet/models/*.keras`, `SignAlphaSet/models/analysis_*/`, `SignAlphaSet/models/*.png`

Action: see Phase 0 in the roadmap. This is a `git filter-repo` / `git-lfs migrate` situation, which rewrites history — **requires explicit sign-off**, not something to do as a drive-by cleanup.

### 1.5 Structural oddity worth flagging, not necessarily fixing

- `main.py` (repo root, Flask+SocketIO, port 8000) and `api/signai_api.py` (Flask, port 5000) are two independent HTTP servers doing overlapping jobs (both take a video and return a translation). `CLAUDE.md` already documents this as intentional/separate, so this isn't "dead code" — but it is duplicate *architecture*, and worth a product decision (is `main.py` still needed, or is it a superseded prototype of `api/signai_api.py`?). Flagging for awareness, not included in the deletion roadmap.
- Root-level clutter: `train.py`, `train-seq2seq.py`, `augemantations.py`, `preprocessing_train_data.py`, `utils_experimental_train.py`, `model.py`, `main.py`, `wlasl/`, `PHOENIX-Weather-2014T/`, `tools/` all sit directly at repo root with no package structure. Addressed in Section 2.

---

## 2. Proposed folder structure

The repo currently has no `src/`-style layout — training, inference, dataset tooling, and three separate applications (desktop app, product website, SignAlphaSet) are flattened into the repo root. Given `SignAlphaSet` and `product_webside` are explicitly independent per `CLAUDE.md`, the goal isn't to force everything into one package — it's to stop the *core DGS pipeline* files from being loose at root, and to group dataset/maintenance scripts that currently have no home.

```
SignAI/
├── signai/                        # NEW: the core DGS training+inference package (currently loose at root)
│   ├── __init__.py
│   ├── training/
│   │   ├── train.py                # from root train.py
│   │   ├── train_seq2seq.py        # from root train-seq2seq.py (drop the hyphen — not importable as-is today)
│   │   ├── model.py                # from root model.py, IF kept (see 1.1 — likely just delete)
│   │   └── experimental_transformer.py  # from utils_experimental_train.py, IF kept
│   ├── augmentation.py             # from augemantations.py (fix the filename typo while moving)
│   └── preprocessing/
│       ├── train_data.py           # from root preprocessing_train_data.py
│       └── normalization.py        # NEW: shared shoulder-centering/scaling logic (see 1.3), imported by
│                                    #      both train_data.py and api/preprocessing_live_data.py
│
├── api/                            # unchanged — already well-scoped (signai_api.py, inference.py,
│                                    #   preprocessing_live_data.py, request.py)
│
├── app/                            # unchanged desktop app, EXCEPT:
│   └── packaging/                  # NEW: consolidate app/builds/*.ifp + specs here, separate from
│                                    #   the .py build scripts (keep app/builds/ for the python scripts only)
│
├── product_webside/                # unchanged — independent Flask site
├── SignAlphaSet/                   # unchanged — independent sub-project
│
├── datasets/                       # NEW: home for one-off dataset acquisition scripts, currently loose at root
│   ├── wlasl/                      # from root wlasl/
│   └── phoenix_weather_2014t/      # from root PHOENIX-Weather-2014T/
│
├── tools/                          # unchanged — already correctly scoped (fix_keras_config.py, etc.)
├── tokenizers/                     # unchanged
├── models/                         # unchanged location, but see Phase 0 for what's inside it
├── data/                           # unchanged
└── logs/                           # unchanged location, but see Phase 0 for what's inside it
```

Notes on restraint (per the "no speculative abstraction" rule — this is sized to the problem, not a green-field redesign):
- **Not** proposing to merge `SignAlphaSet` or `product_webside` into `signai/` — `CLAUDE.md` is explicit that they're intentionally independent, and forcing a shared package would be exactly the kind of premature abstraction to avoid.
- **Not** proposing a `src/` wrapper on top of `signai/` — Python packages don't need it, and `CLAUDE.md`/`AGENTS.md` already document flat root-level entrypoints (`main.py`, `train.py`) that scripts and docs reference; moving those requires updating every doc/command reference, which is real cost for a purely cosmetic gain. If root files move, `CLAUDE.md`, `AGENTS.md`, and `README.md` **must** be updated in the same change (this is already a stated project rule).
- The `normalization.py` extraction is listed here for completeness but should happen only as its own careful, tested change (Section 1.3), not as a side effect of a file-shuffling PR.

---

## 3. Actionable roadmap

Ordered by priority. Each phase is independently shippable — don't block later phases on earlier ones being "perfect."

### Phase 0 — Repo bloat (highest impact, needs your explicit sign-off before running)
1. Confirm which `.keras` / TensorBoard log files in `models/` and `logs/` are still needed for reproducibility vs. safe to move to GitHub Releases / external storage.
2. Move the models you want to keep accessible to GitHub Releases (or an artifact store), update `README.md`/`CLAUDE.md` with the download instructions.
3. Remove the large binaries from git history with `git filter-repo` (or `git lfs migrate`) — **this rewrites history and requires a force-push and everyone re-cloning**, so this step needs your explicit go-ahead and a heads-up to any collaborators before it happens.
4. Add `logs/`, `*.keras` (except intentionally-kept examples), and TensorBoard event files to `.gitignore` if not already fully covered.

### Phase 1 — Safe deletions (no history rewrite, low risk, reversible via git)
1. Delete `model.py`, `preprocessing_train_data_old.py`, `models/doc.txt`.
2. Delete `__pycache__/preprecessing_livedata_web.cpython-310.pyc` and untrack the `__pycache__/` directory (`git rm -r --cached __pycache__`) — confirm `.gitignore` already excludes it going forward (it does per current ignore rules, this is just cleaning up the pre-existing tracked copy).
3. Confirm `utils_experimental_train.py` is genuinely abandoned (quick check-in with you), then delete or relocate.
4. Resolve the `SignAI - Updater.spec` / `SignAI_Updater.spec` duplication — delete the unused one.
5. Rewrite `app/builds/README.md` as a single coherent document instead of two concatenated versions.

### Phase 2 — Structural moves (Section 2's folder layout)
1. Create `signai/` package; move `train.py`, `train-seq2seq.py` → `train_seq2seq.py`, `augemantations.py` → `augmentation.py`, `preprocessing_train_data.py` into it.
2. Update every reference in `CLAUDE.md`, `AGENTS.md`, `README.md`, and any PyInstaller `.spec` `pathex`/hidden-imports that assume root-level file locations.
3. Move `wlasl/` and `PHOENIX-Weather-2014T/` under `datasets/`.
4. Consolidate `.ifp`/`.spec` packaging files under `app/packaging/` once Phase 1's duplicate-spec question is resolved.
5. Re-run `train.py --rebuild-cache` (or equivalent smoke test) and a manual desktop-app launch after the move, to confirm nothing broke on import paths — **verify, don't assume**, per this repo's stated working rules.

### Phase 3 — The one genuinely risky consolidation (do last, own PR, own tests)
1. Extract shared shoulder-centering/normalization logic (Section 1.3) into `signai/preprocessing/normalization.py`.
2. Before switching either `preprocessing_train_data.py` or `api/preprocessing_live_data.py` to use it, snapshot current output on a fixed sample video/CSV and diff numerically against the post-refactor output to confirm bit-for-bit (or float-tolerance) parity.
3. Only merge once parity is confirmed — a silent behavior change here would retrain-vs-inference-skew the whole model without any error being raised.

---

## What this plan deliberately does not do

- Does not touch `product_webside/` or `SignAlphaSet/` internals — no dead code was found there beyond what's already listed (their READMEs/structure looked intentional and in-use).
- Does not propose new abstractions, config systems, or test frameworks beyond what's needed to safely execute the moves above — that would be scope creep beyond "clean up what exists."
- Does not execute anything. Waiting for your go-ahead, especially on Phase 0 (history rewrite) and the two `[UNVERIFIED]` items (updater duplication, `.ifp` files).
