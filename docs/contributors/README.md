# Contributing Guide

## Workflow at a Glance

```text
feature/<user>/<description> → user/<name> → dev → main

documentation/<description> ───────────────→ dev
```

Feature branches are optional. User branches track each contributor's progress
across sprints. Merging into a user branch does not mean the work has passed
team review.

| Branch | Purpose | Accepted PR sources |
|---|---|---|
| `main` | Working, production-ready releases | `dev` |
| `dev` | Shared development | User branches; shared contributor-documentation branches |
| `user/<name>` | Individual contributions | Feature branches; merges from `dev` for synchronisation |
| `feature/<user>/<description>` | One feature or coherent change | A dependent feature may initially target its prerequisite |
| `documentation/<description>` | One shared contributor-guide update, based on `dev` | — |

Changes to `dev` and `main` must go through PRs. Do not push directly.

## Pull Requests

### Choose the Scope

A `user → dev` PR should deliver **one complete PBI or a completed part of one
PBI covering at least one acceptance-criteria checklist item**.

- Identify completed checklist items, the checks that demonstrate them, and
  any remaining items. Link the PBI and sprint.
- Keep unfinished work and unrelated PBIs out of the PR.
- Keep the user branch focused on that delivery until it merges into `dev`.
  Prepare and review other work in separate feature branches while waiting.
- Check the full diff against `dev` before adding another feature: a PR
  includes all pending branch changes, not a selected commit.
- Close the PBI issue only when all its acceptance criteria are met.

### Review Before Merging

| PR route | Approval |
|---|---|
| Feature → user | Author may merge after self-review and relevant checks |
| Documentation → `dev` | At least one teammate other than the author |
| User → `dev` | At least one teammate other than the author |
| `dev` → `main` | At least one teammate other than the author |

Teammates may review feature PRs early, especially when other work depends on
them. GitHub distinguishes merging your own PR from submitting an approving
review; authors cannot submit approving reviews on their own PRs. Repository
permissions and branch rules must also permit the merge.

Before merging any PR:

- Describe the change and link the task or PBI.
- Record relevant checks and their results; checks should pass.
- Resolve review discussions and merge conflicts.
- Obtain another review for changes made after teammate approval.

### Choose the Merge Method

| Source → target | Method |
|---|---|
| Feature without active dependents → user | Squash and merge |
| Feature with active dependents → user | Create a merge commit |
| Documentation → `dev` | Squash and merge |
| User → `dev` | Create a merge commit |
| `dev` → `main` | Create a merge commit |

Squashing combines work-in-progress commits into one logical change. Merge
commits preserve shared history for continuing or dependent branches. GitHub's
“Create a merge commit” uses `git merge --no-ff`.

## Working on Multiple Features

### Independent Features

Start from the updated user branch if it contains no unrelated pending work.
Otherwise, use a clean starting point such as the latest `dev`.

### Dependent Features

1. Branch from the prerequisite feature and initially target its branch with
   the dependent PR. This keeps the PR diff focused on the additional change.
2. When the prerequisite changes, update the dependent branch and rerun checks.
3. Keep ordinary commits while dependents are active; merge the prerequisite
   into the user branch with a merge commit.
4. Update the dependent branch and retarget its PR to the user branch. Confirm
   that the diff contains only its own changes.

If the prerequisite was squash-merged, rebase only the dependent branch's own
commits onto the updated user branch before retargeting. Coordinate with anyone
using published commits before rewriting their history.

## Shared Contributor Documentation

Shared guidance such as `docs/contributors/**` may use a documentation branch
directly into `dev`. Feature-specific documentation follows its feature through
the usual user-branch route.

1. Create `documentation/<description>` from the latest `dev`. Use a separate
   worktree when separating documentation from unfinished feature work.
2. Include one coherent guide update. Supporting documentation-index and README
   navigation changes are allowed; unrelated code, configuration and dependency
   changes are excluded.
3. Check the diff against `dev`, obtain teammate approval, and squash-merge.
4. Merge `dev` into user branches to receive the approved guidance, then archive
   the completed documentation branch.

Documentation reaches `main` through the normal `dev → main` release process.

## Code Readability and Documentation

Write Python and notebook code for readers who may be new to the project or
less familiar with Python conventions.

- Use descriptive names such as `missing_files` so the purpose of a value is
  clear where it is used.
- Make collection checks explicit. For example, use `if len(missing_files) > 0:`
  when checking whether any required files are missing. `if missing_files:`
  relies on Python treating an empty list as false and a non-empty list as true;
  the explicit count makes that condition easier for less experienced readers.
- Explain why a non-obvious operation is needed, including its effect on caller
  input. For example, `Path(data_dir).expanduser()` allows a directory such as
  `~/datasets/fraud`: it expands `~` to the user's home directory because
  `Path()` alone does not. Document this accepted input in the function's
  docstring and explain the reason briefly beside the operation.
- In notebooks, introduce each distinct step with a short Markdown cell that
  explains its purpose and any prerequisite. Give extension activation its own
  documented code cell before cells that depend on its magic commands. Avoid
  prose that merely repeats individual lines of code.

For type annotations, docstrings, and the boundary between reusable package
logic and notebook analysis, see
[Reusable code and Python documentation](../notebook-syncing.md#reusable-code-and-python-documentation).

## Commit Messages

Use `type(scope): summary`; the scope and parentheses are optional.

| Type | Use |
|---|---|
| `feat` | Add functionality |
| `fix` | Correct a defect |
| `docs` | Change documentation |
| `refactor` | Restructure code without changing behaviour |
| `test` | Add or update tests |
| `build` | Change dependencies or build configuration |
| `ci` | Change automated integration workflows |
| `chore` | Other maintenance |

- Use a clear action verb such as “add”, “fix” or “update”; avoid “changes”.
- Add a body when the reason needs explanation, and reference the task or PBI
  issue number when applicable.
- Write squash messages for the completed change, rather than listing checkpoint
  messages. Automatically generated merge messages may keep their defaults.

### Feature-Branch Checkpoints

The message format above is recommended, not required, for intermediate feature
commits. Commit small, understandable checkpoints and label incomplete work
clearly. Checkpoints need not finish a feature or pass every check; the delivered
change must meet the PR review and validation requirements before merging.

## Keeping Branches Current

- **User branches:** merge or rebase the latest `dev`, resolve conflicts, and
  rerun relevant checks. Coordinate before rebasing published commits that
  others depend on.
- **Out-of-date PRs:** update the source from its target before merging, resolve
  conflicts, and rerun checks. Keep the existing PR to preserve review history.

## Archiving Completed Branches

After merging, prefix completed feature and documentation branch names with
`archived/`. Preserve the full original name:

| Original | Archived |
|---|---|
| `feature/spiralsky/add-jupytext` | `archived/feature/spiralsky/add-jupytext` |
| `documentation/contributor-guide` | `archived/documentation/contributor-guide` |

- Update and retarget dependent PRs to the user branch before archiving their
  prerequisite branch.
- Retain archived branches until assessment is complete as contribution evidence;
  do not reuse them for further work.
- Start new features from an updated user branch without unrelated pending work,
  or the latest `dev` as described above. Start documentation branches from `dev`.

## Releases and Repository Settings

- Merge `dev → main` for working, production-ready releases.
- Approved shared contributor-documentation updates may enter `dev` independently
  of feature work and do not require a release tag.
- Tag all sprint and production releases as `v<MAJOR>.<MINOR>.<RELEASE>`.
- Enable merge commits and squash merging in repository settings.
- Leave “Require linear history” disabled for `main`, `dev` and user branches
  because this workflow permits merge commits.
