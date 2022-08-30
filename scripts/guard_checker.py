#!/usr/bin/env python
# Linux only

import os

DIR_PREFIX = 'src/'


def transform_filename(filename):
    return '__' + filename.replace('.', '_').upper() + '__'


def check_file(filename):
    postfix = str.strip(filename.split('.')[-1])
    if postfix != 'h' and postfix != 'hpp':
        return True
    lines = None
    with open(DIR_PREFIX + filename, 'r') as f:
        lines = f.readlines()
    assert lines is not None
    for line_number, line in enumerate(lines):
        line = str.strip(line)
        if line.startswith('#ifndef'):
            guard_name = line.split()[1]
            expected_guard_name = transform_filename(filename)
            if (guard_name != expected_guard_name):
                print(f'{filename}: {guard_name} != {expected_guard_name}')
                return False


def main():
    filenames = [i.name for i in os.scandir(DIR_PREFIX) if i.is_file()]
    for filename in filenames:
        check_file(filename)


if __name__ == '__main__':
    main()
