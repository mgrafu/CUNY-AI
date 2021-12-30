#!/usr/bin/env python
"""Reads a tokenized file to train an output Word2Vec model."""

import argparse
import multiprocessing

from gensim.models import Word2Vec, FastText
from gensim.models.word2vec import LineSentence


def _build_w2v_model(sourcefile: str):
    source = LineSentence(sourcefile)
    model = Word2Vec(
        source, window=15, min_count=10, workers=multiprocessing.cpu_count()
    )
    return model


def main(args: argparse.Namespace) -> None:
    model = _build_w2v_model(args.datasource)
    model.save(args.modelfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasource",
        type=str,
        default="data/en.tok",
        help="link to tokenized data source file",
    )
    parser.add_argument(
        "--modelfile",
        type=str,
        default="data/model.w2v",
        help="output model file name",
    )

    main(parser.parse_args())
