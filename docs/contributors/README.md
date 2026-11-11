# Contributing Guide

## Git

### Workflow Overview
Changes move from optional feature branches into user branches,
then into dev, and finally into main for a release.
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
Only pull requests from user branches are accepted.

#### User
A branch belonging to each contributor within the project. User branches exist
to record and track progress of each contributor during sprints.

**Accepted PR source branches**:
Only pull requests from feature branches are accepted.
Merging dev into a user branch for synchronisation is also allowed, as described
under Keeping Branches Up to Date.

#### Feature
Optional branches for implementing individual features for merging into user branches.

### Pull Requests

#### Scope
Each PBI should be merged individually into dev once complete.
Its pull request should not include unfinished work or changes belonging
to other PBIs.

#### Review and Validation
Every pull request should be approved by at least one team member
other than its author before merging.
The pull request should describe the changes, reference the relevant
task or PBI, and record the checks performed and their results.
Relevant checks should pass, review discussions should be resolved,
and merge conflicts should be resolved before merging.
Changes made after approval should be reviewed before merging.

#### Merge Methods

| Source | Target | Method |
|---|---|---|
| Completed feature branch | Its user branch | Squash and merge |
| User branch | dev | Create a merge commit |
| dev | main | Create a merge commit |

Squash merging combines a feature branch's work-in-progress commits.
Merge commits preserve shared history between continuing branches.
GitHub's “Create a merge commit” corresponds to `git merge --no-ff`.

#### Direct Pushes
Changes to main and dev should be made through pull requests.
Direct pushes to these branches should not be used.

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

### Feature Branch Cleanup
After squash-merging, the feature branch should no longer be used.
Further work should start on a new branch from the updated user branch.

### Releases and Naming
Merges into main should happen per working, production-ready release.
All sprint and production releases should be tagged.
Consistent naming should be used throughout this project.
Release tags should follow the format `v<MAJOR>.<MINOR>.<RELEASE>`.

### Repository Settings
Merge commits and squash merging should be enabled in the repository.
Require linear history should remain disabled for main, dev and user
branches, since this workflow allows merge commits on those branches.
