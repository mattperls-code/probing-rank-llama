import math
from util import tokenize
from collection_statistics import collection_doc_count, collection_term_count, collection_doc_freq_table, collection_term_freq_table
import random
from typing import Callable

feature_names: set[str] = [
    "covered_query_term_count",         # number of query terms that occur in the doc
    "covered_query_term_ratio",         # ratio of query terms that occur in the doc
    "doc_len",                          # number of terms in the document
    "log_doc_len",                      # log of number of terms in the document
    "inverse_doc_len",                  # inverse of number of terms in the document
    "inverse_log_doc_len",              # inverse of log of number of terms in the document
    "query_len",                        # number of terms in the query
    "sum_term_count",                   # sum of number of times a query term appears in the doc
    "avg_term_count",                   # avg of number of times a query term appears in the doc
    "sum_tf",                           # sum of probability a query term appears in the doc
    "avg_tf",                           # avg of probability a query term appears in the doc
    "sum_idf",                          # sum of log of inverse document frequency of a query term if it appears in the doc
    "avg_idf",                          # avg of log of inverse document frequency of a query term if it appears in the doc
    "covered_idf_ratio",                # sum of idf of terms in doc and query divided by sum of idf of terms in query
    "sum_tfidf",                        # sum of product of tf and idf of a query term
    "avg_tfidf",                        # avg of product of tf and idf of a query term
    "bm25",                             # bm25 score
    "log_bm25",                         # log of bm25 score
    "sqrt_bm25",                        # sqrt of bm25 score
    "direchlet_ql",                     # query likelihood score with direchlet smoothing
    "jelinek_mercer_ql",                # query likelihood score with jelinek-mercer smoothing
    "kl_wrt_query",                     # kl divergence using query for true distribution and smoothed doc for approx
    "kl_wrt_doc",                       # kl divergence using doc for true distribution and smoothed query for approx
    "random_noise_10",                  # a random sample on -10 to 10 to detect overfitting
    "random_noise_50",                  # a random sample on -50 to 50 to detect overfitting
]

def covered_query_term_count(doc_terms: list[str], query_terms: list[str]) -> float:
    doc_terms_set = set(doc_terms)

    return len([ query_term for query_term in query_terms if query_term in doc_terms_set ])

def covered_query_term_ratio(doc_terms: list[str], query_terms: list[str]) -> float:
    return covered_query_term_count(doc_terms, query_terms) / len(query_terms)

def doc_len(doc_terms: list[str], query_terms: list[str]) -> float:
    return len(doc_terms)

def log_doc_len(doc_terms: list[str], query_terms: list[str]) -> float:
    return math.log(len(doc_terms) + 1)

def inverse_doc_len(doc_terms: list[str], query_terms: list[str]) -> float:
    return 1 / len(doc_terms)

def inverse_log_doc_len(doc_terms: list[str], query_terms: list[str]) -> float:
    return 1 / math.log(len(doc_terms) + 1)

def query_len(doc_terms: list[str], query_terms: list[str]) -> float:
    return len(query_terms)

def get_freq_table(terms: list[str]) -> dict[str, int]:
    freq_table = {}

    for term in terms: freq_table[term] = freq_table.get(term, 0) + 1

    return freq_table

def sum_term_count(doc_terms: list[str], query_terms: list[str]) -> float:
    doc_freq_table = get_freq_table(doc_terms)

    return sum([ doc_freq_table.get(query_term, 0) for query_term in query_terms ])

def avg_term_count(doc_terms: list[str], query_terms: list[str]) -> float:
    return sum_term_count(doc_terms, query_terms) / len(query_terms)

def sum_tf(doc_terms: list[str], query_terms: list[str]) -> float:
    return sum_term_count(doc_terms, query_terms) / len(doc_terms)

def avg_tf(doc_terms: list[str], query_terms: list[str]) -> float:
    return sum_tf(doc_terms, query_terms) / len(query_terms)

def sum_idf(doc_terms: list[str], query_terms: list[str]) -> float:
    doc_terms_set = set(doc_terms)

    return sum([
        math.log(collection_doc_count / (collection_doc_freq_table.get(query_term, 0) + 1))
        for query_term in query_terms if query_term in doc_terms_set
    ])

def avg_idf(doc_terms: list[str], query_terms: list[str]) -> float:
    return sum_idf(doc_terms, query_terms) / len(query_terms)

def covered_idf_ratio(doc_terms: list[str], query_terms: list[str]) -> float:
    return sum_idf(doc_terms, query_terms) / sum_idf(query_terms, query_terms)

def sum_tfidf(doc_terms: list[str], query_terms: list[str]) -> float:
    doc_len = len(doc_terms)
    doc_terms_set = set(doc_terms)
    doc_freq_table = get_freq_table(doc_terms)

    return sum([
        doc_freq_table[query_term] / doc_len * math.log(collection_doc_count / (collection_doc_freq_table.get(query_term, 0) + 1))
        for query_term in query_terms if query_term in doc_terms_set
    ])

def avg_tfidf(doc_terms: list[str], query_terms: list[str]) -> float:
    return sum_tfidf(doc_terms, query_terms) / len(query_terms)

def bm25(doc_terms: list[str], query_terms: list[str]) -> float:
    K = 1.5
    B = 0.75

    avg_doc_len = collection_term_count / collection_doc_count

    doc_len = len(doc_terms)
    doc_freq_table = get_freq_table(doc_terms)

    score = 0.0

    for query_term in query_terms:
        df = collection_doc_freq_table.get(query_term, 0)
        idf = math.log(1 + (collection_doc_count - df + 0.5) / (df + 0.5))

        tf = doc_freq_table.get(query_term, 0)

        num = idf * tf * (K + 1)
        den = tf + K * (1 - B + B * doc_len / avg_doc_len)

        score += num / den

    return score

def log_bm25(doc_terms: list[str], query_terms: list[str]) -> float:
    return math.log(bm25(doc_terms, query_terms) + 1)

def sqrt_bm25(doc_terms: list[str], query_terms: list[str]) -> float:
    return math.sqrt(bm25(doc_terms, query_terms))

def direchlet_ql(doc_terms: list[str], query_terms: list[str]) -> float:
    MU = 1000

    doc_len = len(doc_terms)
    doc_freq_table = get_freq_table(doc_terms)

    score = 0.0

    for query_term in query_terms:
        num = doc_freq_table.get(query_term, 0) + MU * collection_term_freq_table.get(query_term, 0) / collection_term_count
        den = doc_len + MU

        if num == 0: continue

        score += math.log(num / den)

    return score

def jelinek_mercer_ql(doc_terms: list[str], query_terms: list[str]) -> float:
    LAMBDA = 0.2

    doc_len = len(doc_terms)
    doc_freq_table = get_freq_table(doc_terms)
    
    score = 0.0

    for query_term in query_terms:
        p_doc = doc_freq_table.get(query_term, 0) / doc_len
        p_collection = collection_term_freq_table.get(query_term, 0) / collection_term_count

        p_smoothed = p_doc * (1 - LAMBDA) + p_collection * LAMBDA

        if p_smoothed == 0: continue

        score += math.log(p_smoothed)

    return score

def kl_wrt_query(doc_terms: list[str], query_terms: list[str]) -> float:
    ALPHA = 0.2

    doc_len = len(doc_terms)
    query_len = len(query_terms)
    doc_freq_table = get_freq_table(doc_terms)
    query_freq_table = get_freq_table(query_terms)

    score = 0.0

    for doc_term, doc_term_freq in doc_freq_table.items():
        p_doc = doc_term_freq / doc_len
        p_query = query_freq_table.get(doc_term, 0) / query_len
        p_collection = collection_term_freq_table.get(doc_term, 1) / collection_term_count

        p_query_smoothed = p_query * (1 - ALPHA) + p_collection * ALPHA

        if p_query_smoothed == 0: continue

        score += p_doc * math.log(p_doc / p_query_smoothed)
        
    return score

def kl_wrt_doc(doc_terms: list[str], query_terms: list[str]) -> float:
    return kl_wrt_query(query_terms, doc_terms)

def random_noise_10(doc_terms: list[str], query_terms: list[str]) -> float:
    return random.uniform(-10.0, 10.0)

def random_noise_50(doc_terms: list[str], query_terms: list[str]) -> float:
    return random.uniform(-50.0, 50.0)

compute_feature: dict[str, Callable[[list[str], list[str]], float]] = {
    "covered_query_term_count": covered_query_term_count,         # number of query terms that occur in the doc
    "covered_query_term_ratio": covered_query_term_ratio,         # ratio of query terms that occur in the doc
    "doc_len": doc_len,                          # number of terms in the document
    "log_doc_len": log_doc_len,                      # log of number of terms in the document
    "inverse_doc_len": inverse_doc_len,                  # inverse of number of terms in the document
    "inverse_log_doc_len": inverse_log_doc_len,              # inverse of log of number of terms in the document
    "query_len": query_len,                        # number of terms in the query
    "sum_term_count": sum_term_count,                   # sum of number of times a query term appears in the doc
    "avg_term_count": avg_term_count,                   # avg of number of times a query term appears in the doc
    "sum_tf": sum_tf,                           # sum of probability a query term appears in the doc
    "avg_tf": avg_tf,                           # avg of probability a query term appears in the doc
    "sum_idf": sum_idf,                          # sum of log of inverse document frequency of a query term if it appears in the doc
    "avg_idf": avg_idf,                          # avg of log of inverse document frequency of a query term if it appears in the doc
    "covered_idf_ratio": covered_idf_ratio,                # sum of idf of terms in doc and query divided by sum of idf of terms in query
    "sum_tfidf": sum_tfidf,                        # sum of product of tf and idf of a query term
    "avg_tfidf": avg_tfidf,                        # avg of product of tf and idf of a query term
    "bm25": bm25,                             # bm25 score
    "log_bm25": log_bm25,                         # log of bm25 score
    "sqrt_bm25": sqrt_bm25,                        # sqrt of bm25 score
    "direchlet_ql": direchlet_ql,                     # query likelihood score with direchlet smoothing
    "jelinek_mercer_ql": jelinek_mercer_ql,                # query likelihood score with jelinek-mercer smoothing
    "kl_wrt_query": kl_wrt_query,                     # kl divergence using query for true distribution and smoothed doc for approx
    "kl_wrt_doc": kl_wrt_doc,                       # kl divergence using doc for true distribution and smoothed query for approx
    "random_noise_10": random_noise_10,                  # a random sample on -10 to 10 to detect overfitting
    "random_noise_50": random_noise_50,                  # a random sample on -50 to 50 to detect overfitting
}

def compute_features(doc: str, query: str) -> dict[str, float]:
    doc_terms = tokenize(doc)
    query_terms = tokenize(query)

    return { feature: compute_feature[feature](doc_terms, query_terms) for feature in feature_names }

def main():
    test_docs = [
        """"
            Modern cities face increasing pressure to redesign transportation systems as populations grow and land becomes scarcer. Private car ownership, once seen as a symbol of personal freedom, has introduced congestion, pollution, and inefficient land use in dense urban environments. As a result, many city planners now emphasize public transit, walkability, and mixed-use development as core principles of sustainable growth.
            Rail systems such as subways and light rail offer high passenger capacity with relatively low emissions per rider. Bus rapid transit systems provide a more flexible and lower-cost alternative, especially in cities without existing rail infrastructure. Beyond transit, urban design choices—such as narrower streets, protected bike lanes, and pedestrian plazas—can significantly alter travel behavior by making non-automobile options safer and more attractive.
            Technology also plays a growing role in transportation planning. Real-time traffic monitoring, dynamic toll pricing, and shared mobility platforms allow cities to adapt infrastructure usage to demand. However, these solutions raise concerns about equity, privacy, and access, particularly for lower-income residents. Successful transportation reform therefore requires not only technical efficiency but also careful attention to social outcomes and long-term environmental goals.
        """,
        """
            Machine learning refers to a class of algorithms that identify patterns in data and improve performance through experience rather than explicit programming. At its core, a machine learning model represents a mathematical function that maps inputs to outputs, optimized according to a predefined objective. Common tasks include classification, regression, and clustering.
            Supervised learning relies on labeled data, where the correct output is known in advance. Models such as linear regression, decision trees, and neural networks fall into this category. During training, the model adjusts internal parameters to minimize a loss function that measures prediction error. Unsupervised learning, by contrast, operates without labeled examples and instead seeks structure in the data itself, as seen in techniques like k-means clustering and principal component analysis.
            Model evaluation is a critical step in the machine learning pipeline. Metrics such as accuracy, precision, recall, and cross-entropy quantify performance, but their interpretation depends on the problem context. Overfitting remains a central challenge, occurring when a model captures noise rather than meaningful signal. Regularization, data augmentation, and validation techniques are commonly used to address this issue and ensure models generalize well to unseen data.
        """,
        """
            Oceans play a central role in regulating Earth's climate by absorbing heat and carbon dioxide from the atmosphere. Covering more than seventy percent of the planet's surface, they act as a massive thermal reservoir, moderating temperature extremes between seasons and geographic regions. Without this buffering effect, global climate variability would be far more severe.
            One of the most important mechanisms driving oceanic climate influence is thermohaline circulation, a system of deep-water currents powered by differences in temperature and salinity. These currents transport heat across vast distances, linking tropical and polar regions. Disruptions to this circulation, potentially caused by melting polar ice and freshwater influx, could significantly alter regional climates.
            Oceans also serve as a major carbon sink, absorbing roughly a quarter of human-produced carbon dioxide emissions. While this slows the pace of atmospheric warming, it leads to ocean acidification, which threatens coral reefs, shell-forming organisms, and marine ecosystems more broadly. Understanding ocean dynamics is therefore essential not only for climate science but also for fisheries management, coastal planning, and long-term environmental resilience.
        """
    ]

    test_queries = [
        "the role of light rale systems in transportation",
        "common loss functions in neural networks",
        "effects of climate change on marine life"
    ]

    for test_doc in test_docs:
        for test_query in test_queries:
            features = compute_features(test_doc, test_query)

            print("Features:")
            print("\n".join(f"\t{k}: {v:.4f}" for k, v in features.items()))
            print()

if __name__ == "__main__": main()