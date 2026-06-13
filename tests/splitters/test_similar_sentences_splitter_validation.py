"""Regression tests for ``group_max_sentences`` validation in
``SimilarSentenceSplitter``.

These tests use lightweight fakes for the sentence splitter and the similarity
model so they run fast and offline, without downloading spaCy models or
transformer weights.
"""
import sys
import types

import pytest


# The ``semantic_split`` package eagerly imports ``spacy`` and
# ``sentence_transformers`` in its ``__init__.py``. Those heavy libraries are
# not needed for the grouping logic under test, so we stub them ONLY when they
# are not installed. If the real libraries are present, they are used as-is and
# these stubs are never created.
_stubbed_modules = []


def _ensure_importable(name, **attrs):
    try:
        __import__(name)
    except ImportError:
        module = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        sys.modules[name] = module
        _stubbed_modules.append(name)


_ensure_importable("spacy", load=lambda *args, **kwargs: None)
_ensure_importable(
    "sentence_transformers",
    SentenceTransformer=object,
    util=types.SimpleNamespace(),
)

from semantic_split import SimilarSentenceSplitter, Splitter  # noqa: E402

# Drop the temporary stubs so they do not leak into other test modules. The
# classes under test are already imported/cached and only use plain Python.
for _name in _stubbed_modules:
    sys.modules.pop(_name, None)


class FakeSentenceSplitter(Splitter):
    """Returns a fixed list of sentences and records how often it is called."""

    def __init__(self, sentences):
        self.sentences = list(sentences)
        self.split_calls = 0

    def split(self, text):
        self.split_calls += 1
        return list(self.sentences)


class FakeSimilarityModel:
    """Returns a constant pairwise similarity for deterministic grouping.

    Mirrors ``SentenceTransformersSimilarity``: ``similarities()`` returns one
    score per adjacent sentence pair, i.e. ``len(sentences) - 1`` values.
    """

    def __init__(self, pair_similarity=1.0, similarity_threshold=0.5):
        self.pair_similarity = pair_similarity
        self.similarity_threshold = similarity_threshold
        self.similarities_calls = 0

    def similarities(self, sentences):
        self.similarities_calls += 1
        return [self.pair_similarity] * max(0, len(sentences) - 1)


def make_splitter(sentences, pair_similarity=1.0, similarity_threshold=0.5):
    sentence_splitter = FakeSentenceSplitter(sentences)
    model = FakeSimilarityModel(pair_similarity, similarity_threshold)
    splitter = SimilarSentenceSplitter(
        similarity_model=model, sentence_splitter=sentence_splitter
    )
    return splitter, sentence_splitter, model


@pytest.mark.parametrize("invalid", [0, -1, -5])
def test_invalid_group_max_sentences_raises(invalid):
    splitter, _, _ = make_splitter(["a", "b", "c"])
    with pytest.raises(ValueError):
        splitter.split("ignored", group_max_sentences=invalid)


@pytest.mark.parametrize("invalid", [0, -1, -5])
def test_invalid_group_max_sentences_fails_before_any_work(invalid):
    splitter, sentence_splitter, model = make_splitter(["a", "b", "c"])
    with pytest.raises(ValueError):
        splitter.split("ignored", group_max_sentences=invalid)
    # Fail fast: neither sentence splitting nor embedding should have happened.
    assert sentence_splitter.split_calls == 0
    assert model.similarities_calls == 0


def test_one_sentence_groups_with_valid_value():
    # Every sentence is similar, but group_max_sentences=1 forces singletons.
    sentences = ["a", "b", "c", "d"]
    splitter, _, _ = make_splitter(sentences, pair_similarity=1.0,
                                   similarity_threshold=0.5)
    result = splitter.split("ignored", group_max_sentences=1)
    assert result == [["a"], ["b"], ["c"], ["d"]]


def test_multi_sentence_grouping_with_valid_value():
    # Every sentence is similar, so they pack up to group_max_sentences each.
    sentences = ["a", "b", "c", "d", "e"]
    splitter, _, _ = make_splitter(sentences, pair_similarity=1.0,
                                   similarity_threshold=0.5)
    result = splitter.split("ignored", group_max_sentences=2)
    assert result == [["a", "b"], ["c", "d"], ["e"]]


def test_default_value_groups_all_similar_sentences():
    # Default group_max_sentences=5; six similar sentences -> sizes [5, 1].
    sentences = ["s0", "s1", "s2", "s3", "s4", "s5"]
    splitter, _, _ = make_splitter(sentences, pair_similarity=1.0,
                                   similarity_threshold=0.5)
    result = splitter.split("ignored")
    assert result == [["s0", "s1", "s2", "s3", "s4"], ["s5"]]


def test_empty_input_returns_empty_list():
    splitter, _, model = make_splitter([])
    result = splitter.split("")
    assert result == []
    # No sentences -> no embedding work is needed.
    assert model.similarities_calls == 0
