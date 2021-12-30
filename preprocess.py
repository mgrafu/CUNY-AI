import csv
import nltk
import numpy as np

from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer


def get_duration(st, et):
    minutes = int((et - st) / 60)
    seconds = (et - st) % 60
    duration = f"{minutes} min {seconds:.2f} sec"
    return duration


def check_periods(full_text):
    checked_text = []
    for char in full_text:
        if char in ['.', ',', ':', ';']:
            char = char + " "
        checked_text.append(char)
    return "".join(checked_text)


def preprocess_line(line):
    sent_tokens = [word.casefold() for word in nltk.tokenize.word_tokenize(line)]
    processed_tokens = []
    for word in sent_tokens:
        if word.isalpha():
            processed_tokens.append(word)
        elif word.isnumeric():
            processed_tokens.append("<NUM>")
    return processed_tokens


def preprocess(text):
    all_lines = []
    all_tokens = []
    for line in nltk.tokenize.sent_tokenize(text):
        all_lines.append(line)
        processed_tokens = preprocess_line(line)
        all_tokens.append(processed_tokens)
    return all_lines, all_tokens


def get_all_texts(fraction=1):
    all_titles = []
    all_texts = []
    all_proc_texts = []
    true_or_fake = []
    with open("data/Fake.csv", "r") as csvfile:
        for row in csv.reader(csvfile):
            title, text, _, _ = row
            text = check_periods(text)
            all_titles.append(title)
            all_texts.append(text)
            all_proc_texts.append(preprocess(text))
            true_or_fake.append(1)
    with open("data/True.csv", "r") as csvfile:
        for row in csv.reader(csvfile):
            title, text, _, _ = row
            text = check_periods(text)
            all_titles.append(title)
            all_texts.append(text)
            all_proc_texts.append(preprocess(text))
            true_or_fake.append(0)
    return all_titles, all_texts, all_proc_texts, true_or_fake


def get_clusters():
    clusters = []
    with open("data/clusters.txt", "r") as source:
        for line in source:
            clusters.append(int(line.rstrip()))
    return clusters


def get_word_embedding(word, model):
    sw = stopwords.words('english')
    try:
        word_emb = model.wv[word]
    except:
        word_emb = 0
    if word in sw:
        word_emb = word_emb * 0.5
    return word_emb


def get_sent_embedding(sent_tokens, model):
    sent_embedding = np.zeros([100,])
    for word in sent_tokens:
        tok_emb = get_word_embedding(word, model=model)
        sent_embedding += tok_emb
    return sent_embedding


def get_text_embedding(prep_text, model):
    text_embedding = np.zeros([100,])
    for sent in prep_text:
        sent_emb = get_sent_embedding(sent, model=model)
        text_embedding += sent_emb
    return text_embedding


def get_bow(texts_list, sw="english"):
    return CountVectorizer(stop_words=sw).fit_transform(texts_list).toarray()
