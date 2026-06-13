"""
Regression tests for group_max_sentences validation in SimilarSentenceSplitter.

Uses lightweight fake sentence splitters and similarity models so that no
spaCy models or transformer weights need to be downloaded.
"""

import pytest
from semantic_split.splitters.SimilarSentenceSplitter import SimilarSentenceSplitter
from semantic_split.splitters.Splitter import Splitter
from typing import List


class FakeSentenceSplitter(Splitter):
    def __init__(self, sentences):
        self._sentences = sentences
        self.split_called = False

    def split(self, text):
        self.split_called = True
        return list(self._sentences)


class FakeSimilarityModel:
    def __init__(self, similarity_value=0.9, similarity_threshold=0.5):
        self.similarity_threshold = similarity_threshold
        self._value = similarity_value
        self.call_count = 0

    def similarities(self, sentences):
        self.call_count += 1
        return [self._value] * (len(sentences) - 1)


def _make_splitter(sentences, similarity=0.9, threshold=0.5):
    return SimilarSentenceSplitter(
        similarity_model=FakeSimilarityModel(similarity, threshold),
        sentence_splitter=FakeSentenceSplitter(sentences),
    )
