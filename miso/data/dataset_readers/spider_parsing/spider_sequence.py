# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict, Counter
from typing import List, Any
import pdb 
import copy 
import re 
import json
import numpy as np

import spacy
from spacy.language import Language
import en_core_web_sm

import networkx as nx
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok

from transformers import AutoTokenizer


class SpiderSequence:
    def __init__():