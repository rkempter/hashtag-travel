"""
Renato Kempter, 2014
renato.kempter@gmail.com

This modules handles the topic generation and assignes topics to clusters
"""

import json
import logging
import os

from collections import defaultdict, Counter
from instagram_collector.collector import connect_postgres_db
from gensim import corpora, models

import pandas as pd

logging.basicConfig(
    filename='topic_logs.log',level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def clean_tags(conn, query):
    """
    Loads the hashtags from database and cleans them. Tags that appear only once in the corpus
    are removed. Returns the filtered documents
    :param conn: The database connection
    :param query: The query to load data from database
    :return: list of cleaned documents
    """
    logging.info("Start cleaning tags")

    df_hashtags = pd.read_sql(query, conn)

    # extract all tags
    docs = df_hashtags['tags'].str.split(',').values

    # flatten the list
    hashtags_flat = [tag for subtags in docs for tag in subtags if tag != '']

    # Count all hashtags with
    hashtags_count = Counter(tag for tag in hashtags_flat)

    # Filter out hashtags that appear only once
    filtered_hashtag_all = set([hashtag for hashtag, count in hashtags_count.items() if count == 1])

    documents = []

    for doc in docs:
        new_doc = [
            hashtag
            for hashtag in doc
            if hashtag not in filtered_hashtag_all and hashtag != ''
        ]
        if new_doc:
            documents.append(new_doc)

    logging.info("Done with cleaning the tags")
    return documents


def generate_topics(documents, store_path, nbr_topics=100):
    """
    Takes a number of documents and generates nbr_topics topics. For persistency, the model
    can be saved to disc
    :param documents:
    :param store_path:
    :param nbr_topics:
    """
    logging.info("Start generating topics")
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(document) for document in documents]

    # Generate a tf idf model
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    topic_model = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=nbr_topics)

    dictionary.save(os.path.join(store_path, "dictionary.dict"))
    topic_model.save(os.path.join(store_path, "model.lda"))
    logging.info("Done generating topics")

def load_model(store_path):
    """
    Loads a model from disk
    :param store_path:
    :return: gensim.models.LdaModel
    """
    logging.info("Loading an lda model at %s" % store_path)
    return models.LdaModel.load(os.path.join(store_path, "model.lda"))


def load_dictionary(store_path):
    """
    Loads a dictionary from disk
    :param store_path:
    :return: corpora.Dictionary
    """
    logging.info("Loading dictionary at %s" % store_path)
    return corpora.Dictionary.load(os.path.join(store_path, "dictionary.dict"))


def load_tfidf_model(store_path):
    """
    Loads a tfidf model
    :param store_path:
    :return: gensim.models.TfidfModel
    """
    logging.info("Loading the tfidf model at %s" % store_path)
    return models.TfidfModel.load(os.path.join(store_path, "model.tfidf"))


def get_topics(lda_model, documents):
    """
    Generates a topic distribution for a number of documents
    :param documents: tfidf document corpus
    :param store_path:
    :return:
    """
    logging.info("Start generating the topics for documents")

    corpus_model = lda_model[documents]

    topic_distribution = defaultdict(float)

    # compute distribution
    for document in corpus_model:
        for topic, probability in document:
            topic_distribution[topic] += probability

    # normalize
    total = sum(topic_distribution.values())

    logging.info("Done generating the topics for documents")

    return {key: (val / total) for key, val in topic_distribution.items()}


def get_cluster_topic_distribution(conn, store_path):
    """
    Compute the distribution of topics in the clusters
    :param conn:
    :param store_path: Path were all models and dictionary are stored
    :return: json document of topic distribution in clusters
    """
    logging.info("Start generating topic distribution for clusters")
    query = """
        SELECT
            cluster_id, tags
        FROM
            media_events
        WHERE
            cluster_id IS NOT NULL AND
            tags != '';"""

    df_tags = pd.read_sql(query, conn)

    grouped = df_tags.groupby('cluster_id')['tags']

    cluster_topic_distribution = {}

    lda_model = load_model(store_path)
    dictionary = load_dictionary(store_path)
    tfidf_model = load_model(store_path)

    for name, group in grouped:
        corpus = [dictionary.doc2bow(document) for document in group.str.split(',').values]
        cluster_topic_distribution[name] = get_topics(lda_model, tfidf_model[corpus])

    return json.dumps(cluster_topic_distribution)


def get_topic_names(store_path, threshold=0.05, topic_number=100):
    """
    Get the tags that stand for a topic. Only use tags that have higher confidence than threshold
    :param store_path: Path were models and dictionaries are stored
    :param threshold: A threshold to pick only tags that have higher confidence than this number
    :param topic_number: Number of topics we have
    :return:
    """
    logging.info("Retrieve topic names")
    lda_model = load_dictionary(store_path)

    topic_names = {}

    for topic_index in range(0, topic_number):
        topic_words = lda_model.show_topic(topic_index)
        names = [word for probability, word in topic_words if probability > threshold]

        if names:
            topic_names[topic_index] = names

    return json.dumps(topic_names)


if __name__ == '__main__':
    start_query = """
        SELECT tags
        FROM media_events
        WHERE tags != '';"""
    connection = connect_postgres_db()
    storage_path = "/home/rkempter/"

    training_documents = clean_tags(connection, start_query)
    generate_topics(training_documents, storage_path)

    cluster_distribution = get_cluster_topic_distribution(connection, storage_path)
    topics = get_topic_names(storage_path)

