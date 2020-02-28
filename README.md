# WikiFactCheck-en

This repository contains the data to accompany 
'WikiFactCheck-en: Automated Fact-Checking of Claims from Wikipedia'.
  
Contents as follows:
```
.
│
├── wikifactcheck-en_full0.jsonl
├── wikifactcheck-en_full1.jsonl
├── wikifactcheck-en_full2.jsonl
├── wikifactcheck-en_full3.jsonl
├── wikifactcheck-en_full4.jsonl
│
├── wikifactcheck-en_test.jsonl
└── wikifactcheck-en_train.jsonl
```

As explained in the paper, the annotated portion of the corpus is split into `train` and `test` sets.
The entirety of the data (including annotated as well as non-annotated) is contained in the `full` sets, split into 5 for space constraints.

You may want to make use of the provided loading script to make use of the
dataset in your code.
```
usage: loadwfc-en.py [-h] [-d] [-f]
                     [-r [{train,test,full} [{train,test,full} ...]]]
                     [-n NUMLINES] [-t {json,python}]

optional arguments:
  -h, --help            show this help message and exit
  -d, --download        download dataset
  -f, --force           force re-download?
  -r [{train,test,full} [{train,test,full} ...]], --read [{train,test,full} [{train,test,full} ...]]
                        read from particular datasets (default: all)
  -n NUMLINES, --numlines NUMLINES
                        numlines to read from each one
  -t {json,python}, --fmt {json,python}
                        output format for --read option
```
