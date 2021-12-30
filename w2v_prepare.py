#!/usr/bin/env python
"""Writes a preprocessed data file from a GZ file url."""

import argparse
import gzip
import urllib.request

import nltk


def _prepare(url: str, sinkfile: str) -> None:
    with urllib.request.urlopen(url) as response, gzip.GzipFile(
        fileobj=response
    ) as uncompressed, open(sinkfile, "w") as sink:
        for line in uncompressed:
            tokens = nltk.tokenize.word_tokenize(line.decode("utf-8"))
            sentence = []
            for word in tokens:
                if word.isalpha():
                    sentence.append(word.casefold())
                elif word.isnumeric():
                    sentence.append("<NUM>")
            sentence = " ".join(sentence)
            print(sentence, file=sink)


def main(args: argparse.Namespace) -> None:
    _prepare(args.datasource_url, args.tokenfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasource_url",
        type=str,
        default="http://data.statmt.org/news-crawl/en/news.2007.en.shuffled.deduped.gz",  # noqa: E501
        help="link to data source GZ file",
    )
    parser.add_argument(
        "--tokenfile",
        type=str,
        default="en.tok",
        help="output token file name",
    )

    main(parser.parse_args())
