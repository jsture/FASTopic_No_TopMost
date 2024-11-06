import numpy as np
from fastopic_no_topmost.topmost.gensim import STOPWORDS
import logging
import os
import json


def get_top_words(beta, vocab, num_top_words, verbose=False):
    topic_str_list = list()
    for i, topic_dist in enumerate(beta):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][
            : -(num_top_words + 1) : -1
        ]
        topic_str = " ".join(topic_words)
        topic_str_list.append(topic_str)
        if verbose:
            print("Topic {}: {}".format(i, topic_str))

    return topic_str_list


def get_stopwords_set(stopwords=[]):
    return STOPWORDS


class Logger:
    def __init__(self, level):
        self.logger = logging.getLogger("TopMost")
        self.set_level(level)
        self._add_handler()
        self.logger.propagate = False

    def info(self, message):
        self.logger.info(f"{message}")

    def warning(self, message):
        self.logger.warning(f"WARNING: {message}")

    def set_level(self, level):
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level in levels:
            self.logger.setLevel(level)

    def _add_handler(self):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
        self.logger.addHandler(sh)

        # Remove duplicate handlers
        if len(self.logger.handlers) > 1:
            self.logger.handlers = [self.logger.handlers[0]]


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def read_text(path):
    texts = list()
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            texts.append(line.strip())
    return texts


def save_text(texts, path):
    with open(path, "w", encoding="utf-8") as file:
        for text in texts:
            file.write(text.strip() + "\n")


def read_jsonlist(path):
    data = list()
    with open(path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            data.append(json.loads(line))
    return data


def save_jsonlist(list_of_json_objects, path, sort_keys=True):
    with open(path, "w", encoding="utf-8") as output_file:
        for obj in list_of_json_objects:
            output_file.write(json.dumps(obj, sort_keys=sort_keys) + "\n")


def split_text_word(texts):
    texts = [text.split() for text in texts]
    return texts
