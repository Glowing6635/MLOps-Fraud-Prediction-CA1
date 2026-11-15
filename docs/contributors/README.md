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
| Feature → user, including features with active dependents | Squash and merge |
| Documentation → `dev` | Squash and merge |
| User → `dev` | Create a merge commit |
| `dev` → `main` | Create a merge commit |

Squash each feature into one logical commit when integrating it into a user
branch. Active dependents do not change this merge method; rebase them as
explained below. User-to-dev and dev-to-main integrations retain merge commits;
there is no later squash step. GitHub's “Create a merge commit” uses
`git merge --no-ff`.

## Working on Multiple Features

### Independent Features

Start from the updated user branch if it contains no unrelated pending work.
Otherwise, use a clean starting point such as the latest `dev`.

### Dependent Features

1. Branch from the prerequisite feature and initially target its branch with
   the dependent PR. This keeps the PR diff focused on the additional change.
2. When the prerequisite changes, update the dependent branch and rerun checks.
3. Before squash-merging the prerequisite into the user branch, record its
   original tip SHA. Keep that history available until all dependents are updated.
4. After the squash merge, rebase only each dependent branch's additional
   commits onto the updated user branch. Preserve unfinished work, including
   untracked files, before rebasing. Run the rebase in that branch's worktree
   if it is checked out elsewhere.
5. Retarget the dependent PR to the user branch. Check that its diff contains
   only its own feature changes, and rerun relevant checks.

For example, while on a dependent branch, replace `OLD_PREREQUISITE_SHA` with
the recorded prerequisite tip and run:

```bash
git rebase --onto user/spiralsky OLD_PREREQUISITE_SHA
git diff user/spiralsky...HEAD
```

Changing the PR target alone does not remove the original prerequisite commits.
If the dependent branch has no additional commits, rebasing simply moves it onto
the updated user branch. Coordinate before rewriting published dependent history;
when updating its remote branch, use `git push --force-with-lease` rather than an
unconditional force push. Do not force-push shared user, dev, or main branches
as part of this procedure.

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
