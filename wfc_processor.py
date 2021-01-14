# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""  """


import os
import re
from tqdm import tqdm

from transformers.utils import logging
from .utils import DataProcessor, InputExample, InputFeatures
import datasets

from colorama import Fore, Back, Style
from blessings import Terminal; T = Terminal()

logger = logging.get_logger(__name__)


class WfcProcessor(DataProcessor):
    """Processor for the WFC dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self, cache_dir='/sw/mcs/wfc'):#, language, train_language=None):
#         self.language = language
#         self.train_language = train_language
        pass
            
    def get_examples(self, split, data_dir, both=True):
        
        print('INFO:', 'get_examples', split, f'both={both}')
        
        lines = self._read_tsv(os.path.join(data_dir, "wfc_%s.tsv" % split))
        examples = []
        for i, line in tqdm(enumerate(lines)):
            if i == 0: # header
                continue
            elif i >= 100:
                return examples
            try:
                (ix, url, context, id_, refuted, claim, md5) = line
            except ValueError:
                continue
                
            guid = "%s-%s" % ("train", i)
            with open('./utf-refdata/'+md5, 'r') as e:
                evidence = re.sub(r"[\n\t\s]+", ' ', e.read())
                evidence = evidence.split('.')
#             text_a = line[0]
#             text_b = line[1]
            
            if both or i % 2: # if not both, then pick entailed only on odd indices
                examples.append(InputExample(guid=guid+'e', text_a=claim, text_b=evidence, label='supported'))
            if both or (i+1) % 2:
                examples.append(InputExample(guid=guid+'r', text_a=refuted, text_b=evidence, label='refuted'))
        return examples
    
    def get_train_examples(self, data_dir, both=True):
        """See base class."""
        return self.get_examples('train', data_dir, both=both)
    
    def get_test_examples(self, data_dir, both=True):
        """See base class."""
        return self.get_examples('dev', data_dir, both=both)
    
    def get_labels(self):
        """See base class."""
        return ["supported", "refuted"]


wfc_processors = {
    "wfc": WfcProcessor,
}

wfc_output_modes = {
    "wfc": "classification",
}

wfc_tasks_num_labels = {
    "wfc": 2,
}


def wfc_convert_examples_to_features(
    examples, # claim, list[evidence] pair
    tokenizer,
    max_length=None,#: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = wfc_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = wfc_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample):# -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

#     batch_encoding = tokenizer(
#         [(example.text_a, example.text_b) for example in examples],
#         max_length=max_length,
#         padding="max_length",
#         truncation=True,
#     )
    
    all_encodings = [
        tokenizer(
            [(example.text_a, text_b) for text_b in example.text_b],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        ) 
        for example in examples
    ]

    features = []
    for i in range(len(examples)):
        this_features = []
        for j in range(len(examples[i].text_b)): # iterating over sentences in evidence
            inputs = {k: all_encodings[i][k][j] for k in all_encodings[i].keys()}
            
            feature = InputFeatures(**inputs, label=labels[i])
            this_features.append(feature)
            
        features.append(this_features)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example %d ***" % i)
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i][:5])

    return features