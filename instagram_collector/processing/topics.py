"""
Renato Kempter, 2014
renato.kempter@gmail.com

This modules handles the topic generation and assignes topics to clusters
"""

import json
import logging
import numpy as np
import os
import pandas as pd
import subprocess

from collections import defaultdict, Counter
from decimal import Decimal
from instagram_collector.collector import connect_postgres_db
from gensim import corpora, models
from operator import itemgetter
from pymongo import MongoClient

from .config import TOPIC_NBR, BTM_CALL

logging.basicConfig(
    filename='topic_logs.log',level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def clean_tags(conn, query, stop_words=[]):
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
    hashtags_flat = [tag for subtags in docs for tag in subtags
                     if tag != '' and len(tag) > 3 and tag not in stop_words]

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
        if new_doc and len(new_doc) >= 2:
            documents.append(new_doc)

    logging.info("Number of unique hashtags: %d" % len(filtered_hashtag_all))
    #logging.info("Mean of hashtags per document: %f" % float(sum(map(lambda x: len(x), documents))) / len(documents))
    logging.info("Done with cleaning the tags")
    return documents


def generate_btm_topics(documents, store_path, topic_collection, cluster_collection,
                        alpha, beta, niter, save_step, nbr_topics=TOPIC_NBR):
    """
    Biterm topic modeling, using the software from the following:
    https://code.google.com/p/btm/ and the corresponding paper
    http://www2013.wwwconference.org/proceedings/p1445.pdf
    :param documents: { cluster_id: x, tokens: [] }
    :param store_path:
    :param nbr_topics:
    :return:
    """
    # First we need to create the dictionary

    dictionary = corpora.Dictionary(documents).token2id
    input_path = os.path.join(store_path, "btm_input.txt")
    doc2cluster = []
    with open(input_path, "w+") as token_doc_file:
        for document in documents:
            print dictionary.token2id
            token_doc = map(lambda token: dictionary.token2id[token], document["tokens"])
            token_doc_file.write("%s\n" % " ".join(token_doc))
            doc2cluster.append(document["cluster_id"])

    # Learn paramters p(z) and p(z|w)
    return_code = subprocess.call([BTM_CALL, "est", nbr_topics, len(dictionary.token2id),
                                   alpha, beta, niter, save_step, input_path, store_path])

    if return_code:
        raise ValueError("Wrong return code while estimating parameters")

    # Infer p(z|d)
    return_code = subprocess.call([BTM_CALL, "inf", "lda", nbr_topics, input_path, store_path])

    if return_code:
        raise ValueError("Wrong return code received while infering p(z|d)")

    return doc2cluster


def write_mongo_btm_topics(topic_collection, store_path, threshold=0.01, topic_number=TOPIC_NBR):
    """
    Write the topics to the mongodb
    :param topic_collection:
    :param store_path:
    :param threshold:
    :param topic_number:
    :return:
    """
    topics = []

    dictionary = load_dictionary(store_path).id2token
    topic_nbr = 0
    with open(os.path.join(store_path, "pw_z.k%d" % topic_nbr)) as topic:
        for distribution in topic.readlines():
            distribution = distribution.split()
            values = [Decimal(value) for value in distribution]
            word2value = zip(range(len(values)), values)
            word2value_sorted = sorted(word2value, key=itemgetter(1), reverse=True)

            topics.append({
                "_id": topic_nbr,
                "words": [(dictionary[token[0]], token[1]) for token in word2value_sorted[:15]],
                "names": [dictionary[token[0]] for token in word2value_sorted[:15] if token[1] > threshold],
                "clusters": [],
            })
            topic_nbr += 1
    topic_collection.insert(topics)


def write_btm_cluster_vector(cluster_collection, store_path, doc2cluster_map, topic_nbr=TOPIC_NBR):
    """
    Read in the output from the external btm program and generate the cluster vectors
    :param cluster_collection:
    :param store_path:
    :param doc2cluster_map:
    :param topic_nbr:
    :return:
    """
    clusters = {}
    document_id = 0
    with open(os.path.join(store_path, "pz_d.k%d" % topic_nbr)) as document_collection:
        for document_vector in document_collection.readlines():
            topic_values = np.array([Decimal(value) for value in document_vector.split()])
            cluster_id = doc2cluster_map[document_id]

            if cluster_id in clusters:
                clusters[cluster_id] += topic_values
            else:
                clusters[cluster_id] = topic_values

    for cluster_id, vector in clusters.items():
        vector_normalized = vector / np.sum(vector)

        cluster_collection.update({"_id": cluster_id},
                                  {"$set": {"distribution": [str(val) for val in vector_normalized.tolist()]}})


def generate_topics(documents, store_path, nbr_topics=TOPIC_NBR, tfidf_on=False):
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
    if tfidf_on:
        tfidf = models.TfidfModel(corpus)
        corpus = tfidf[corpus]
    topic_model = models.LdaModel(corpus, id2word=dictionary, num_topics=nbr_topics)

    dictionary.save(os.path.join(store_path, "dictionary.dict"))
    topic_model.save(os.path.join(store_path, "model.lda"))
    logging.info("Done generating topics")


def write_mongo_topics(topic_collection, store_path, threshold=0.05, topic_number=TOPIC_NBR):
    topics = []

    lda_model = load_model(store_path)
    for topic_index in range(0, topic_number):
        topic_words = lda_model.show_topic(topic_index)
        names = [word for probability, word in topic_words if probability > threshold]

        if not names:
            names = [topic_words[0][1]]

        topics.append({
            "_id": topic_index,
            "words": topic_words,
            "names": names,
            "clusters": []
        })

    return topic_collection.insert(topics)


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

    topic_distribution = defaultdict(Decimal)

    # compute distribution
    for document in corpus_model:
        for topic, probability in document:
            topic_distribution[topic] += probability

    # normalize
    total = sum(topic_distribution.values())

    logging.info("Done generating the topics for documents")

    return {"%d" % key: str((val / total)) for key, val in topic_distribution.items()}


def write_mongodb_distribution(conn, store_path, cluster_collection):
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

    lda_model = load_model(store_path)
    dictionary = load_dictionary(store_path)
    tfidf_model = load_model(store_path)

    for name, group in grouped:
        corpus = [dictionary.doc2bow(document) for document in group.str.split(',').values]
        distribution = get_topics(lda_model, tfidf_model[corpus])
        cluster_collection.update({"_id": name},
                                  {"$set": {"distribution": distribution}},
                                  upsert=False)


def get_topic_names(store_path, threshold=0.05, topic_number=TOPIC_NBR):
    """
    Get the tags that stand for a topic. Only use tags that have higher confidence than threshold
    :param store_path: Path were models and dictionaries are stored
    :param threshold: A threshold to pick only tags that have higher confidence than this number
    :param topic_number: Number of topics we have
    :return:
    """
    logging.info("Retrieve topic names")
    lda_model = load_model(store_path)

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

    client = MongoClient('localhost', 27017)

    # call paris database in mongo db
    mongo_db = client.paris_db
    mongo_db.topic_collection.remove({})
    training_documents = clean_tags(connection, start_query)
    generate_topics(training_documents, storage_path)

    write_mongo_topics(mongo_db.topic_collection, storage_path)
    write_mongodb_distribution(connection, storage_path, mongo_db.cluster_collection)
