# coding=utf-8
"""System monitoring functions."""
from __future__ import print_function, division
from future.utils import string_types

import datetime

import os
from collections import OrderedDict
from subprocess import check_output

import psutil

import numpy as np
import pandas as pd


# --- nvidia GPUs
# in loopy we are using py-nvml, so maybe go that way

def float_or_nan(val):
    try:
        return float(val)
    except ValueError:
        return np.nan


def int_or_neg(val):
    try:
        return int(val)
    except ValueError:
        return -1


# Define known resources data types and human-friendly order
_COL2TYPE = OrderedDict([
    ('pid', ('pid', int_or_neg)),
    ('gpu', ('gpu', int_or_neg)),
    ('type', ('computational', lambda x: x == 'C')),
    ('sm', ('sm_pct', float_or_nan)),
    ('mem', ('mem_pct', float_or_nan)),
    ('enc', ('enc_pct', float_or_nan)),
    ('dec', ('dec_pct', float_or_nan)),
    ('fb', ('mem_MB', float_or_nan)),
    ('command', ('command', str)),
    ('pwr', ('power_W', float_or_nan)),
    ('temp', ('temp_C', float_or_nan)),
    ('mclk', ('mclk_MHZ', float_or_nan)),
    ('pclk', ('pclk_MHZ', float_or_nan)),
    ('pviol', ('pviol_pct', float_or_nan)),
    ('tviol', ('tviol_pct', float_or_nan)),
    ('bar1', ('bar1_MB', float_or_nan)),
    ('sbecc', ('sbecc_errs', float_or_nan)),
    ('dbecc', ('dbecc_errs', float_or_nan)),
    ('pci', ('pci_errs', float_or_nan)),
    ('rxpci', ('rxpci_MB/s', float_or_nan)),
    ('txpci', ('txpci_MB/s', float_or_nan)),
])


def _human_order(columns):

    # Process-specific columns
    pinfos = set(k for k in columns if not k.startswith('card_'))
    # Card-specific columns
    cinfos = set(k for k in columns if k.startswith('card_'))

    # Ordering constraints
    desired_order = [col for col, _ in _COL2TYPE.values()]

    # First time...
    order = ['time']
    # ...then process info
    order += [k for k in desired_order if k in pinfos]
    order += [k for k in pinfos if k not in order]
    # ...then cards info
    order += [('card_' + k) for k in desired_order if ('card_' + k) in cinfos]
    order += [k for k in cinfos if k not in order]

    assert len(order) == len(columns)

    return order


def _humanize_dict(d):
    return OrderedDict((k, d[k]) for k in _human_order(d.keys()))


def _humanize_dataframe(df):
    return df[_human_order(df.columns)]


def nvidia_cards():
    """
    Returns basic information about the nvidia cards present in the system.
    A list of tuples (gpu_id, model, uuid).
    """
    def parse_card_info(card_info):
        # GPU 0: GeForce GTX 1060 (UUID: GPU-1da8867e-c3b2-ac55-7f11-9a847dd8ec37)
        gpu_id, _, rest = card_info.partition(':')
        gpu_id = int(gpu_id.split()[1])
        gpu_model, _, rest = rest.partition('(')
        gpu_model = gpu_model.strip()
        uuid = rest[:-1].partition(': ')[2].strip()
        return gpu_id, gpu_model, uuid
    return [parse_card_info(card_info)
            for card_info in check_output(['nvidia-smi', '-L']).splitlines()]


def gpu_processes_info():
    """
    Returns the GPU utilization stats of all the processes using the GPU.

    The return value is a dictionary {(pid, gpu): pinfo}, where pinfo is in turn a dictionary
    containing things like:
      - pid: the process id
      - gpu: the gpu id
      - mem_MB: the memory used by the process, in MB
      - sm_pct: the percentage of cuda cores used by the process
      - card_power_W: the power drawn by the card
      - ...and many others
    """

    #
    # Alternatives for querying nvida GPUs state from python (as of 2017/01/20):
    #
    #   - Good old psutil
    #     No GPU stats: https://github.com/giampaolo/psutil/issues/526
    #
    #   - Official but oldie bindings (ctypes) to nvml (BSD):
    #     https://pypi.python.org/pypi/nvidia-ml-py/
    #     Modernization "forks":
    #       https://github.com/fbcotter/py3nvml
    #       https://github.com/jonsafari/nvidia-ml-py
    #
    #   - nvidia-smi wrapper, nagios plugin, dump all to XML:
    #     loopb.system.NvidiaStats
    #     https://github.com/FastSociety/nagios-nvidia-smi-plugin
    #
    #   - nvidia-smi wrapper, dump selection to CSV:
    #     https://github.com/wookayin/gpustat
    #
    #   - simplemost approaches / the web
    #     https://gist.github.com/matpalm/9c0c7c6a6f3681a0d39d
    #     http://unix.stackexchange.com/questions/252590/how-to-log-gpu-load
    #     http://stackoverflow.com/questions/8223811/top-command-for-gpus-using-cuda
    #     https://www.microway.com/hpc-tech-tips/nvidia-smi_control-your-gpus/
    #
    # For simplicity we can just parse all the output of nvidia-smi, as we are not
    # running this to sample at short intervals. Go for nvml bindings or more concrete
    # queries if performance ever becomes important.
    #
    # Here we do not do anything of the above, but parse the output of "dmon" and "pmon"
    # commands. This currently works with my setup and needs.
    # We could move to xml output, or --query_whatever=whatever --format=csv
    # That would make parsing less dependent on output, and use only 1 call,
    # leading hopefully to more atomic measurement. But it is not obvious to me
    # how to find the utilization of the GPU by a process by other means using
    # nvidia-smi (maybe --query-accounted-apps, but it needs extra setup).
    #
    # For GPU memory utilisation of a model, each library will or will not provide
    # means to specifically query it. The measurements from nvidia-smi might not be
    # trustworthy, as some libraries (tensorflow, I'm looking at you) might decide
    # to reserve more resources than actually needed.
    #

    def normalize(d):
        normalized = {}
        for key, (value, units) in d.items():
            try:
                new_name, vp = _COL2TYPE[key]
                normalized[new_name] = vp(value)
            except KeyError:
                normalized[key + '_' + units] = value
        return normalized

    def parse_nvidia_smi_xmon(output):
        lines = output.splitlines()
        col_names = lines[0][1:].split()
        col_units = lines[1][1:].split()
        rows = [
            normalize({key: (value, units)
                       for key, units, value in
                       zip(col_names, col_units, line.split())})
            for line in lines[2:]
        ]
        return rows

    def index_by(records, by):
        def key(record):
            return tuple(record[k] for k in by) if isinstance(by, tuple) else record[by]
        return {key(record): record for record in records}

    pmon_output = check_output(['nvidia-smi', 'pmon', '-c', '1', '-s', 'um'])
    dmon_output = check_output(['nvidia-smi', 'dmon', '-c', '1', '-s', 'pucvmet'])

    now = datetime.datetime.utcnow()
    pmon_pinfos = index_by(parse_nvidia_smi_xmon(pmon_output), by=('pid', 'gpu'))
    card_infos = index_by(parse_nvidia_smi_xmon(dmon_output), by='gpu')

    # Remove invalid entries
    # (e.g. output from pmon that do not correspond to a process)
    pmon_pinfos = {(pid, gpu): v for (pid, gpu), v in pmon_pinfos.items()
                   if pid >= 0}

    # Merge/rename, add timestamp
    for record in pmon_pinfos.values():
        record['time'] = now
        card_info = card_infos[record['gpu']]
        for k, v in card_info.items():
            if k not in ('gpu',):
                record['card_' + k] = v

    # Human-friendly ordering or records
    pmon_pinfos = {pid: _humanize_dict(record)
                   for pid, record in pmon_pinfos.items()}

    return pmon_pinfos


def gpu_resources(pids=None, recursive=False, as_dataframe=False):
    """
    Returns a dictionary or dataframe with the GPU resources used by some processes.
    Only processes from `pids` that do use the gpu will be listed.

    The processes information come from  a dictionary like that returned by
    `gpu_processes_stats`.

    Parameters
    ----------
    pids : int, list of ints or None, default None
      The pid of the processes we are interested in.
      If None, use current process pid.

    recursive : bool, default False
      If True, find also GPU resources used by pids children.

    as_dataframe : bool, default False
      If True, returns a pandas dataframe with indexed by (pid, gpu); None if no process uses the GPUs
      If False, returns a dictionary {(pid, gpu): gpu_process_info}; empty if no process uses the GPUs
    """
    if pids is None:
        pids = [os.getpid()]
    if not isinstance(pids, list):
        try:
            pids = list(pids)
        except TypeError:
            pids = [pids]
    if recursive:
        # noinspection PyTypeChecker
        for pid in pids:
            pids += psutil.Process(pid).children(recursive=True)
    pinfos = {(pid, gpu): pinfo
              for (pid, gpu), pinfo in gpu_processes_info().items()
              if pid in pids}
    if not as_dataframe:
        return pinfos
    if pinfos:
        df = pd.DataFrame(pinfos.values()).set_index(['pid', 'gpu'], drop=False)
        return _humanize_dataframe(df)
    return None


def total_gpu_resources(pids=None, recursive=True,
                        resources=('sm_pct', 'mem_MB')):
    """
    Returns a pandas dataframe with aggregated values for resources per card.
    If no process is using any resources from the card, returns an empty dataframe.

    Parameters
    ----------
    pids : int, list of ints or None, default None
      The pid of the processes we are interested in.
      If None, use current process pid.

    recursive : bool, default False
      If True, find also GPU resources used by pids children.

    resources : string list, default ('sm_pct', 'mem_MB')
      Names of the resources to aggregate. They should have numerical values.
    """
    # Allow to pass single strings
    if isinstance(resources, string_types):
        resources = resources,

    # Query
    df = gpu_resources(pids=pids, recursive=recursive, as_dataframe=True)

    # No process -> empty dataframe
    if df is None:
        return pd.DataFrame(columns=['gpu'] + list(resources))

    # Define aggregators: sum for process-specific, identity for the rest
    def res2agg(resource):
        if resource.startswith('card_') or resource == 'time':
            return lambda x: x.iloc[0]
        return np.sum
    aggregators = {resource: res2agg(resource) for resource in resources}

    # Aggregate and return
    return df.groupby('gpu').agg(aggregators)[list(resources)]


def _gpu_resources_example():
    from threading import Thread
    from time import sleep
    import torch

    def use_the_gpu(size_MB=512, sleep_interval_s=1, repeats=1000):

        nelems = int(size_MB * 1024 ** 2 / 4)
        tensor = torch.cuda.FloatTensor(nelems).cuda().zero_()

        for _ in range(repeats):
            torch.dot(tensor, tensor)
            if sleep_interval_s is not None:
                sleep(sleep_interval_s)

    thread = Thread(target=use_the_gpu)
    thread.daemon = True
    thread.start()
    sleep(1)
    print(gpu_resources(recursive=True, as_dataframe=True))
    print(gpu_resources())
    # noinspection PyTypeChecker
    print(total_gpu_resources(resources=['sm_pct', 'card_sm_pct']))
    print(nvidia_cards())


if __name__ == '__main__':
    _gpu_resources_example()
