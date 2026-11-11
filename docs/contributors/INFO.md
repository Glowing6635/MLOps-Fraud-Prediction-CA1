# Contributing Information

## Git

### Branches

#### Main Branch
The *main branch* to be used in git.

**PR Requirements** describe what branches pull requests from are accepted. **Merge Requirements** describe requirements for merging such as whether PRs are needed or not.

**PR Requirements**:
Only pull requests from the *dev branch* is accepted. 

**Merge Requirements**:
Merges should happen per *working, production-ready release*.

**Other Info**:
The main branch should only contain working versions of the product and have tagged releases.

#### Dev Branch
The branch for development.

**PR Requirements**:
Only Pull Requests from the *user branch* are accepted.

**Merge Requirements**:
Merges should happen per PBI completion.

#### User Branch
A branch belonging to each contributor within the project. User branches exist to record and track progress of each contributor during sprints.

**PR Requirements**:
Only Pull Requests from feature branches are accepted.

**Merge Requirements**:
None

#### Feature Branches
Feature branches optionally existing for implementing singular features for merging into user branches.

**PR Requirements**:
None

**Merge Requirements**
None

### Git Practices

#### Staying Up-to-date to upstream
User Branches should attempt to stay up-to-date and not be behind in commits to dev branch by rebasing.
