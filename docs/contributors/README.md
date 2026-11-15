# Contributing Guide

## Git

### Workflow Overview
Changes move from optional feature branches into user branches,
then into dev, and finally into main for a release.
Contributors may merge their own feature-to-user pull requests without
teammate approval. User-to-dev pull requests require teammate approval;
merging into a user branch does not mean the work has passed team review.
Shared contributor documentation may instead use temporary documentation
branches with approved pull requests directly to dev, as described below.
The source branch supplies the changes in a pull request;
the target branch receives them.

### Branch Roles
**Accepted PR source branches** identifies which branches may supply changes
through pull requests to the branch being described. Shared review and merge
rules are defined under Pull Requests.

#### Main
The main branch should only contain working versions of the product.

**Accepted PR source branches**:
Only pull requests from the dev branch are accepted.

#### Dev
The branch for development.

**Accepted PR source branches**:
Pull requests from user branches are accepted for development work.
Pull requests from documentation branches are also accepted when limited
to shared contributor documentation, as described under Contributor Documentation.

#### User
A continuing branch for each contributor, used across sprints to track
individual progress. Link pull requests to their PBI and sprint so
contributions can be traced.

**Accepted PR source branches**:
Only pull requests from feature branches are accepted.
Merging dev into a user branch for synchronisation is also allowed, as described
under Keeping Branches Up to Date.

#### Feature
Optional branches for implementing individual features for merging into user branches.
Start from the updated user branch when it contains no unrelated pending work.
For independent features, use a starting point without unrelated pending work,
such as the latest dev.
For dependent features, branch from the prerequisite and initially target
the prerequisite branch so the PR shows only the dependent change.
When the prerequisite changes, update the dependent branch and rerun relevant
checks. After the prerequisite merges, update the dependent branch and retarget
its PR to the user branch so it contains only its own changes.

#### Documentation
Temporary branches named `documentation/<description>` hold one coherent
update to shared contributor guidance. Create them from the latest dev
and open pull requests targeting dev.

### Pull Requests

#### Scope
A user-to-dev pull request should deliver either a complete PBI or a completed,
reviewable part of one PBI covering at least one acceptance-criteria checklist
item. Identify the completed items and the checks that demonstrate them, and
state which items remain. Partial delivery does not mean the whole PBI is done;
do not close its issue until all acceptance criteria are met.
The pull request should not include unfinished work or changes belonging
to other PBIs. Keep each user branch focused on that delivery until it is
merged into dev; other work can proceed in separate feature branches.
Before merging another feature into the user branch, check the full diff
against dev: a user-to-dev PR includes all pending changes on the user branch,
not a selected commit. Independent feature branches can be prepared and
reviewed while an earlier delivery awaits approval.

#### Contributor Documentation
Shared contributor guidance, such as `docs/contributors/**`, may be updated
through a temporary `documentation/<description>` branch created from the
latest dev, with a pull request targeting dev. Supporting documentation
index entries and README navigation links may be included.
This is an exception to the usual feature-to-user-to-dev route.
Contributor-documentation updates reach main through the normal
dev-to-main release process.
Feature-specific documentation should follow the feature through dev.

Each documentation pull request must contain only the intended documentation
changes and must be approved by at least one team member other than its author.
All other review and validation requirements still apply. Keep each branch
focused on one coherent update.

To separate documentation from unfinished feature work, create the documentation
branch in a separate worktree based on dev and apply only the intended
changes. Review the diff against dev to confirm that implementation changes
and unrelated configuration or dependency changes are excluded.

After the documentation pull request is merged into dev, contributors should
merge dev into their user branches to receive the updated guidance.
Completed documentation branches should follow the archiving and retention rules below.

#### Review and Validation
Feature-to-user pull requests may be merged by their author without teammate
approval after self-review and relevant checks. Teammates may still review
these PRs early, especially when other features depend on them.
Documentation-to-dev, user-to-dev and dev-to-main pull requests should be
approved by at least one team member other than their author before merging.
Author merging and an approving review are separate actions: GitHub does not
allow authors to submit an approving review on their own PRs. Repository
permissions and branch rules must also allow the chosen merge action.
The pull request should describe the changes, reference the relevant
task or PBI, and record the checks performed and their results.
Relevant checks should pass, review discussions should be resolved,
and merge conflicts should be resolved before merging.
Changes made after teammate approval should be reviewed again before merging.

#### Merge Methods

| Source | Target | Method |
|---|---|---|
| Completed feature branch without active dependent branches | Its user branch | Squash and merge |
| Feature branch with active dependent branches | Its user branch | Create a merge commit |
| Documentation branch | dev | Squash and merge |
| User branch | dev | Create a merge commit |
| dev | main | Create a merge commit |

Squash merging combines a feature branch's work-in-progress commits.
Keep ordinary commits while dependent branches are being developed. Use a
merge commit when landing a prerequisite with active dependent branches to
preserve their shared history. If the prerequisite is squash-merged instead,
rebase only the dependent branch's own commits onto the updated user branch
before retargeting its PR; coordinate before rewriting published history.
Merge commits preserve shared history between continuing branches.
GitHub's “Create a merge commit” corresponds to `git merge --no-ff`.

#### Direct Pushes
Changes to main and dev should be made through pull requests.
Direct pushes to these branches should not be used.

### Commit Messages
Commit messages should use the format `type(scope): summary`.
The scope is optional; omit it and its parentheses when unnecessary.

| Type | Use |
|---|---|
| feat | Add functionality |
| fix | Correct a defect |
| docs | Change documentation |
| refactor | Restructure code without changing its behaviour |
| test | Add or update tests |
| build | Change dependencies or build configuration |
| ci | Change automated integration workflows |
| chore | Other maintenance |

The summary should describe the change clearly, using an action verb
such as "add", "fix" or "update". Avoid vague messages such as "changes".
Add a body when the reason for the change needs further explanation.
Reference the relevant task or PBI issue number when applicable.

Squash commit messages should describe the combined change rather than
list its intermediate work-in-progress messages.
Automatically generated merge commits may retain their default messages.

#### Feature-Branch Commits (Guidelines)
This section simply serves as guidelines for contributors on intermediate
commits on feature branches. The commit-message format above is recommended,
not required, for these commits.

Small, frequent commits may be used to record development checkpoints.
Each commit should focus on one understandable change where practical.
Messages should describe what changed, even when the work is incomplete.
Incomplete checkpoints should be identified clearly in the message.

Intermediate commits do not need to represent a finished feature or
pass every check. The completed change must satisfy the review and
validation requirements before its pull request is merged.

Before squash-merging, the final commit message should be written to
describe the completed change and follow the commit-message format above.

### Keeping Branches Up to Date

#### User Branches
User branches should be kept up to date with the latest changes from dev
by merging or rebasing.
Conflicts should be resolved and relevant checks should be re-run after updating.
Merging dev into a user branch for synchronisation is allowed,
in addition to accepting feature-branch pull requests.
Before rebasing published commits, coordinate with anyone whose work
depends on those commits.

#### Out-of-date Pull Requests
If a pull request falls behind its target branch, update the source branch
with the latest changes from the target branch before merging.
Resolve any conflicts and rerun relevant checks. Keep the existing pull
request so its discussion and review history are preserved.

### Completed Branches
After merging, completed feature and documentation branches should
be renamed by prefixing their full original names with `archived/` and
should no longer be used for further work. Preserve the branch type,
contributor name where present, and descriptive part of the name:

- `feature/spiralsky/add-jupytext` becomes
  `archived/feature/spiralsky/add-jupytext`.
- `documentation/contributor-guide` becomes
  `archived/documentation/contributor-guide`.

Before archiving a prerequisite feature branch, update and retarget any
dependent pull requests to the user branch.

Further feature work should start on a new branch from the updated user branch.
Further shared contributor-documentation work should start on a new branch
from the updated dev branch.
Retain archived branches until assessment is complete so their commits
remain available as evidence of individual contributions.

### Releases and Naming
Merges from dev into main should happen per working, production-ready release.
Approved shared contributor-documentation updates may merge into dev
independently of feature work and do not require a release tag.
All sprint and production releases should be tagged.
Consistent naming should be used throughout this project.
Release tags should follow the format `v<MAJOR>.<MINOR>.<RELEASE>`.

### Repository Settings
Merge commits and squash merging should be enabled in the repository.
Require linear history should remain disabled for main, dev and user
branches, since this workflow allows merge commits on those branches.
