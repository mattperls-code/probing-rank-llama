import ir_datasets
from collections import Counter
from itertools import islice
from util import tokenize

collection = ir_datasets.load("msmarco-passage/train")

# subset selection

use_collection_subset = True
subset_collection_size = 1000000

# desired collection wide stats

collection_doc_count = subset_collection_size if use_collection_subset else collection.docs_count()
collection_term_count = 0

collection_doc_freq_table = Counter()
collection_term_freq_table = Counter()

# process chosen subset

doc_iter = islice(collection.docs_iter(), subset_collection_size) if use_collection_subset else collection.docs_iter()

print("Starting Collection Statistics Calculation")

for doc in doc_iter:
    doc_terms = tokenize(doc.text)

    collection_doc_freq_table.update(set(doc_terms))
    collection_term_freq_table.update(doc_terms)

    collection_term_count += len(doc_terms)

print("Finished Collection Statistics Calculation")
