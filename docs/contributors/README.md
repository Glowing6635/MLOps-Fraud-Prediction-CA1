# Contributing Guide

## Git

### Workflow Overview
Changes move from optional feature branches into user branches,
then into dev, and finally into main for a release.
Shared contributor documentation may instead use temporary documentation
branches with approved pull requests directly to main, as described below.
The source branch supplies the changes in a pull request;
the target branch receives them.

### Branch Roles
**Accepted PR source branches** identifies which branches may supply changes
through pull requests to the branch being described. Shared review and merge
rules are defined under Pull Requests.

#### Main
The main branch should only contain working versions of the product.

**Accepted PR source branches**:
Pull requests from dev are accepted for releases.
Pull requests from documentation branches are also accepted when limited
to shared contributor documentation, as described under Contributor Documentation.

#### Dev
The branch for development.

**Accepted PR source branches**:
Pull requests from user branches are accepted for development work.
Pull requests from main are also accepted to synchronise approved
contributor-documentation updates into dev.

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
Start from the updated user branch unless the feature depends on unfinished
work in another feature branch. For dependent features, merge the prerequisite
first, then update the dependent branch so its PR contains only its own changes.

#### Documentation
Temporary branches named `documentation/<description>` hold one coherent
update to shared contributor guidance. Create them from the latest main
and open pull requests targeting main.

### Pull Requests

#### Scope
Each PBI should be merged individually into dev once complete.
Its pull request should not include unfinished work or changes belonging
to other PBIs.
Keep each user branch focused on one PBI until it is merged into dev;
other work can proceed in separate feature branches.

#### Contributor Documentation
Shared contributor guidance, such as `docs/contributors/**`, may be updated
through a temporary `documentation/<description>` branch created from the
latest main, with a pull request targeting main. Supporting documentation
index entries and README navigation links may be included.
This is an exception to the usual feature-to-user-to-dev-to-main route.
Feature-specific documentation should follow the feature through dev.

Each documentation pull request must contain only the intended documentation
changes and must be approved by at least one team member other than its author.
All other review and validation requirements still apply. Keep each branch
focused on one coherent update.

To separate documentation from unfinished feature work, create the documentation
branch in a separate worktree based on main and apply only the intended
changes. Review the diff against main to confirm that implementation changes
and unrelated configuration or dependency changes are excluded.

After the documentation pull request is merged, open an approved synchronisation
pull request from main to dev. Contributors should then merge dev into their
user branches to receive the updated guidance. Completed documentation branches
should follow the archiving and retention rules below.

#### Review and Validation
Every pull request should be approved by at least one team member
other than its author before merging.
This includes feature-to-user, user-to-dev, dev-to-main,
documentation-to-main and main-to-dev synchronisation pull requests.
The pull request should describe the changes, reference the relevant
task or PBI, and record the checks performed and their results.
Relevant checks should pass, review discussions should be resolved,
and merge conflicts should be resolved before merging.
Changes made after approval should be reviewed before merging.

#### Merge Methods

| Source | Target | Method |
|---|---|---|
| Completed feature branch | Its user branch | Squash and merge |
| Documentation branch | main | Squash and merge |
| main (documentation synchronisation) | dev | Create a merge commit |
| User branch | dev | Create a merge commit |
| dev | main | Create a merge commit |

Squash merging combines a feature branch's work-in-progress commits.
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
After squash-merging, completed feature and documentation branches should
be renamed by prefixing their full original names with `archived/` and
should no longer be used for further work. Preserve the branch type,
contributor name where present, and descriptive part of the name:

- `feature/spiralsky/add-jupytext` becomes
  `archived/feature/spiralsky/add-jupytext`.
- `documentation/contributor-guide` becomes
  `archived/documentation/contributor-guide`.

Further feature work should start on a new branch from the updated user branch.
Further shared contributor-documentation work should start on a new branch
from the updated main branch.
Retain archived branches until assessment is complete so their commits
remain available as evidence of individual contributions.

### Releases and Naming
Merges from dev into main should happen per working, production-ready release.
Approved shared contributor-documentation updates may merge independently
and do not require a release tag.
All sprint and production releases should be tagged.
Consistent naming should be used throughout this project.
Release tags should follow the format `v<MAJOR>.<MINOR>.<RELEASE>`.

### Repository Settings
Merge commits and squash merging should be enabled in the repository.
Require linear history should remain disabled for main, dev and user
branches, since this workflow allows merge commits on those branches.
