# Overview
Repository for team COWY Wildfire Meteorlogical Data at the NSF ASCEND and NVIDIA hackathon.

# Getting Started
1. from home dir, `git clone git@github.com:northlh/2026_nvhackathon.git`
2. Create a conda env: `conda create --name 2026_nvhackathon python=3.11`
3. Run the command: `conda activate 2026_nvhackathon`
4. cd into the repo cloned in 1.
5. prior to running pip below, make sure the branch is correct (install from main!)
6. Install the repo and its dependencies by running: `pip install -e .` (or `pip install -e .[ml]` to also install ML stack, see .toml)

# Other tips
1. To create a branch, cd into the repo, then: `git checkout -b new_branch_name`
