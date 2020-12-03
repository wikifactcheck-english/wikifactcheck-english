# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""WikiFactCheck-English: The realistic context-aware fact-checking dataset."""

from __future__ import absolute_import, division, print_function

import json
import logging
import os
import re

from tqdm.notebook import tqdm

import datasets


_CITATION = """\
@inproceedings{sathe-etal-2020-automated,
    title = "Automated Fact-Checking of Claims from {W}ikipedia",
    author = "Sathe, Aalok  and
      Ather, Salar  and
      Le, Tuan Manh  and
      Perry, Nathan  and
      Park, Joonsuk",
    booktitle = "Proceedings of the 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://www.aclweb.org/anthology/2020.lrec-1.849",
    pages = "6874--6882",
    abstract = "Automated fact checking is becoming increasingly vital as both truthful and fallacious information accumulate online. Research on fact checking has benefited from large-scale datasets such as FEVER and SNLI. However, such datasets suffer from limited applicability due to the synthetic nature of claims and/or evidence written by annotators that differ from real claims and evidence on the internet. To this end, we present WikiFactCheck-English, a dataset of 124k+ triples consisting of a claim, context and an evidence document extracted from English Wikipedia articles and citations, as well as 34k+ manually written claims that are refuted by the evidence documents. This is the largest fact checking dataset consisting of real claims and evidence to date; it will allow the development of fact checking systems that can better process claims and evidence in the real world. We also show that for the NLI subtask, a logistic regression system trained using existing and novel features achieves peak accuracy of 68{\%}, providing a competitive baseline for future work. Also, a decomposable attention model trained on SNLI significantly underperforms the models trained on this dataset, suggesting that models trained on manually generated data may not be sufficiently generalizable or suitable for fact checking real-world claims.",
    language = "English",
    ISBN = "979-10-95546-34-4",
}
"""

_DESCRIPTION = """\
WikiFactCheck-English, a dataset of 124k+ triples consisting of a claim,
context and an evidence document extracted from English Wikipedia articles
and citations, as well as 34k+ manually written claims that are refuted by
the evidence documents. This is the largest fact checking dataset consisting
of real claims and evidence to date; it will allow the development of fact
checking systems that can better process claims and evidence in the real world.
"""


class WikiFactCheckConfig(datasets.BuilderConfig):
    """BuilderConfig for WFC."""

    def __init__(self, **kwargs):
        """BuilderConfig for WFC.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(WikiFactCheckConfig, self).__init__(**kwargs)


class WikiFactCheck(datasets.GeneratorBasedBuilder):
    """WikiFactCheck-English"""

    __PROJECT = 'wikifactcheck-english'
    __BASEDIR = '~/.{}'.format(__PROJECT)
    __REPOURL = 'https://rawcdn.githack.com/{prj}/{prj}/master/'.format(prj=__PROJECT)
    _URL = __REPOURL#"https://github.com/wikifactcheck-english/wikifactcheck-english"
    _DEV_FILE = "wfc_dev.tsv"
    _TRAINING_FILE = "wfc_train.tsv"

    BUILDER_CONFIGS = [
        WikiFactCheckConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "claim": datasets.Value("string"),
                    # "refuted": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "evidence": datasets.Value("string"),
                    "label": datasets.ClassLabel(num_classes=2, 
                                                 names=['supported', 'refuted'])
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            # supervised_keys=None,
            homepage="https://github.com/wikifactcheck-english/wikifactcheck-english",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls_to_download = {
            "train": os.path.join(self._URL, self._TRAINING_FILE),
            "dev": os.path.join(self._URL, self._DEV_FILE),
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        print('INFO:', downloaded_files)
        
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
        ]

    def _generate_examples(self, filepath, refsdir='utf-refdata', split=None):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            for i, line in tqdm(enumerate(f)):
                if i == 0: continue
                    
                errors = 0
                try:
                    ix, url, context, id_, refuted, claim, evidence_file = line.split('\t')
                except ValueError:
                    print('ERROR: skipping line.', line)
                    errors += 1
                    continue
                    
                print(f'INFO: encountered {errors} errors processing {filepath}')
                    
                evidence_path = refsdir + '/' + evidence_file.strip()
                with open(evidence_path, 'r') as e:
                    evidence = re.sub(r"[\n\t\s]*", ' ', e.read())
                    
                # yield two training examples (supported and refuted) from this
                # datapoint.
                
                # first, the supported claim
                yield id_, {
                    "claim": claim,
                    "label": 'supported',
                    "context": context,
                    "evidence": evidence,
                    "id": id_,
                }
                # next, the refuted claim
                yield id_, {
                    "claim": refuted,
                    "label": 'refuted',
                    "context": context,
                    "evidence": evidence,
                    "id": id_,
                }
                    
#             for article in squad["data"]:
#                 title = article.get("title", "").strip()
#                 for paragraph in article["paragraphs"]:
#                     context = paragraph["context"].strip()
#                     for qa in paragraph["qas"]:
#                         question = qa["question"].strip()
#                         id_ = qa["id"]

#                         answer_starts = [answer["answer_start"] for answer in qa["answers"]]
#                         answers = [answer["text"].strip() for answer in qa["answers"]]

#                         # Features currently used are "context", "question", and "answers".
#                         # Others are extracted here for the ease of future expansions.
#                         yield id_, {
#                             "claim": title,
#                             "refuted": title,
#                             "context": context,
#                             "evidence": EvIdEnCe,
#                             "id": id_,
#                         }