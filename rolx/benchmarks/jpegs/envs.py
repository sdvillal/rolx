from __future__ import print_function, division

from subprocess import CalledProcessError

import os.path as op
import delegator
import json
from ruamel import yaml


def conda_env_info():
    import os
    prefix = os.environ.get('CONDA_PREFIX')
    env = os.environ.get('CONDA_DEFAULT_ENV')
    python_exe = os.environ.get('CONDA_PYTHON_EXE')
    # Others
    # prompt_modifier = os.environ.get('CONDA_PROMPT_MODIFIER=')
    return env, prefix, python_exe


def env2prefix():
    env_prefixes = json.loads(delegator.run('conda env list --json').out)['envs']
    return {op.basename(prefix): prefix for prefix in env_prefixes}


def env_name(env_yaml):
    with open(env_yaml, 'rt') as reader:
        return yaml.safe_load(reader)['name']


def create_environments(environment_yamls, copy_from=None, update='update'):
    import delegator
    for yaml in environment_yamls:
        print('Creating environment %s' % yaml)
        if copy_from is not None:
            # quick and dirty logic added because my plane will take off soon
            res = delegator.run('conda env remove --yes -n %s' % env_name(yaml))
            if res.return_code and 'EnvironmentLocationNotFound' not in res.err:
                raise CalledProcessError(res.return_code, res.cmd, res.err)
            res = delegator.run('conda create --name %s --clone %s' % (env_name(yaml), copy_from))
            if res.return_code:
                raise CalledProcessError(res.return_code, res.cmd, res.err)
            res = delegator.run('conda env update -f %s' % yaml)
            if res.return_code:
                raise CalledProcessError(res.return_code, res.cmd, res.err)
        else:
            res = delegator.run('conda env create -f %s' % yaml)
        if res.return_code:
            if 'prefix already exists' in res.err:
                if update == 'force':
                    print('Forcing creating environment %s' % yaml)
                    delegator.run('conda env create -f %s --force' % yaml)
                elif update == 'prune':
                    print('Updating + pruning environment %s' % yaml)
                    delegator.run('conda env update -f %s --prune' % yaml)
                elif update == 'update':
                    print('Updating environment %s' % yaml)
                    delegator.run('conda env update -f %s' % yaml)
                elif not update:
                    pass
                else:
                    raise CalledProcessError(res.return_code, res.cmd, res.err)
            else:
                raise CalledProcessError(res.return_code, res.cmd, res.err)
