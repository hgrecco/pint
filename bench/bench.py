

from __future__ import division, unicode_literals, print_function, absolute_import

import fnmatch
import os
import copy
from timeit import Timer

import yaml


def time_stmt(stmt='pass', setup='pass', number=0, repeat=3):
    """Timer function with the same behaviour as running `python -m timeit `
    in the command line.

    :return: elapsed time in seconds or NaN if the command failed.
    :rtype: float
    """

    t = Timer(stmt, setup)

    if not number:
        # determine number so that 0.2 <= total time < 2.0
        for i in range(1, 10):
            number = 10**i

            try:
                x = t.timeit(number)
            except:
                print(t.print_exc())
                return float('NaN')

            if x >= 0.2:
                break

    try:
        r = t.repeat(repeat, number)
    except:
        print(t.print_exc())
        return float('NaN')

    best = min(r)

    return best / number


def build_task(task, name='', setup='', number=0, repeat=3):
    nt = copy.copy(task)

    nt['name'] = (name + ' ' + task.get('name', '')).strip()
    nt['setup'] = (setup + '\n' + task.get('setup', '')).strip('\n')
    nt['stmt'] = task.get('stmt', '')
    nt['number'] = task.get('number', number)
    nt['repeat'] = task.get('repeat', repeat)

    return nt


def time_task(name, stmt='pass', setup='pass', number=0, repeat=3, stmts='', base=''):

    if base:
        nvalue = time_stmt(stmt=base, setup=setup, number=number, repeat=repeat)
        yield name + ' (base)', nvalue
        suffix = ' (normalized)'
    else:
        nvalue = 1.
        suffix = ''

    if stmt:
        value = time_stmt(stmt=stmt, setup=setup, number=number, repeat=repeat)
        yield name, value / nvalue

    for task in stmts:
        new_task = build_task(task, name, setup, number, repeat)
        for task_name, value in time_task(**new_task):
            yield task_name + suffix, value / nvalue


def time_file(filename, name='', setup='', number=0, repeat=3):
    """Open a yaml benchmark file an time each statement,

    yields a tuple with filename, task name, time in seconds.
    """
    with open(filename, 'r') as fp:
        tasks = yaml.load(fp)

    for task in tasks:
        new_task = build_task(task, name, setup, number, repeat)
        for task_name, value in time_task(**new_task):
            yield task_name, value


def recursive_glob(rootdir='.', pattern='*'):
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames
            if fnmatch.fnmatch(filename, pattern)]

def main(filenames=None):
    if not filenames:
        filenames = recursive_glob('.', 'bench_*.yaml')
    elif isinstance(filenames, basestring):
        filenames = [filenames, ]

    for filename in filenames:
        print(filename)
        print('-' * len(filename))
        print()
        for task_name, value in time_file(filename):
            print('%.2e   %s' % (value, task_name))
        print()

main()
