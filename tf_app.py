#!/usr/bin/python3
import os
import pathlib
import git

if not os.path.exists('models'):
    git.Repo.clone_from('https://github.com/tensorflow/models.git', 'models')
g = git.Git('models')
g.pull('origin','master')
