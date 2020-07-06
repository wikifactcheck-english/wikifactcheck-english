#!/usr/bin/env python3

import os
import json
from urllib.request import urlopen
from glob import glob
from tqdm import tqdm
from pathlib import Path
from functools import partial
from argparse import ArgumentParser

PROJECT = 'wikifactcheck-english'
BASEDIR = '~/.{}'.format(PROJECT)
REPOURL = 'https://rawcdn.githack.com/{prj}/{prj}/master/'.format(prj=PROJECT)


def download(full=True, dest=BASEDIR, force=False):
    '''
    '''
    parent = Path(dest).expanduser()
    parent.mkdir(exist_ok=True)

    filename = 'wikifactcheck-english_{}.jsonl'
    url = REPOURL + filename
    for part in tqdm(['train', 'test'], desc='train and test'):
        filepath = parent / filename.format(part)
        if filepath.exists() and not force: 
            print(filepath, 'already exists')
            continue
        with filepath.open('wb+') as fp, urlopen(url.format(part)) as web:
            for line in web:
                fp.write(line)

    # if full data is requested (default), also download the 5 full{} part files
    # and combine them into one, also deleting the part files
    if full:
        filename = 'wikifactcheck-english_full{}.jsonl'

        url = REPOURL + filename
        filepath = parent / filename.format('')
        if filepath.exists() and not force: 
            print(filepath, 'already exists')
            return
        with filepath.open('wb+') as combined:
            for part in tqdm(range(5), desc='full'):
                with urlopen(url.format(part)) as web:
                    for line in web:
                        combined.write(line) 


def load_(pattern, lines=None, path=BASEDIR):
    '''
    load from all .jsonl files containing {pattern}
    returns a generator over entries of the appropriate kind
    '''
    parent = Path(path).expanduser()
    # parent.mkdir(exist_ok=True)
    if not parent.exists():
        path = '.'
        parent = Path(path)

    filenames = sorted(glob('*{}*.jsonl'.format(pattern)))
    if not filenames:
        inp = input('Data not found at {}. Download? [Y/n]'.format(path)) 
        if inp.lower() in ('', 'y', 'yes', 'yep', 'yeah'):
            download(dest=path, full='full' in pattern)
        else:
            raise StopIteration

    ctr = 0
    for fname in filenames:
        with Path(fname).open('r') as f:
            for line in f:
                ctr += 1
                yield json.loads(line[:-1])
                if lines and ctr >= lines:
                    raise StopIteration

load = load_
load_train = partial(load_, 'train') # train set
load_test  = partial(load_, 'test')  # held-out test set
load_full  = partial(load_, 'full')  # full dataset, including non-annot.


if __name__ == '__main__':
    parser = ArgumentParser('wikifactcheck-english')
    parser.add_argument('-d', '--download', help='download dataset',
                        action='store_true', default=False)
    parser.add_argument('-f', '--force', help='force re-download?',
                        action='store_true', default=False)
    datasets = ['train', 'test', 'full']
    parser.add_argument('-r', '--read', type=str, nargs='*',
                        choices=datasets,
                        help='read from particular datasets (default: all)')
    parser.add_argument('-n', '--numlines', type=int, default=None,
                        help='numlines to read from each one')
    parser.add_argument('-t', '--fmt', help='output format for --read option',
                        default='json', choices=['json', 'python'])
    args = parser.parse_args()

    if args.download:
        download(force=args.force)
    if args.read:
        for name in args.read:
            ctr = 0
            for item in load(name):
                if args.numlines and ctr >= args.numlines:
                    break
                if args.fmt is 'json': 
                    print(json.dumps(item))
                else:
                    print(item)
                ctr += 1

