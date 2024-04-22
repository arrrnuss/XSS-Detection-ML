import os
import re
import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import ray
import asyncio


from tqdm import tqdm
from collections import Counter
from more_itertools import chunked, flatten
from urllib.parse import urlparse, quote, unquote, parse_qs
from math import log2
from collections import Counter

from parallel_compute import execute_with_ray

warnings.filterwarnings('ignore')
sns.set_theme(font_scale=2)

SEED = 0

NUMBERS = set("0123456789")
SPEC_CHARS = set(["+", '"', "*", "#", "%", "&", "(", ")", "=", "?", "^", "-", ".", "!", "~", "_", ">", "<"])
ALL_CHARS = set(string.printable) - set('\x0b\f\r')



# Function to calculate entropy
def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

# Function to create a set of dictionary words
def create_dictionary_words() -> set[str]:
    df = pd.read_csv('listofattack.csv')
    dictionary_words: set[str] = set(df['Word'].apply(str))
    return dictionary_words

# Main function to enrich rows in chunks
async def enrich_content_chunk(chunk, dictionary_words):
    enriched_rows = []
    for row in chunk:
        tagg = str(row[0])
        
        # Calculate total_chars directly
        total_chars = len(tagg)
        num_spaces = tagg.count(' ')
        num_parenthesis = tagg.count('(') + tagg.count(')')
        num_slash = tagg.count('/')
        num_plus = tagg.count('+')
        num_point = tagg.count('.')
        num_comma = tagg.count(',')
        num_semicolon = tagg.count(';')
        num_alpha = len(re.findall(re.compile(r"\w"), tagg))
        num_numeric = len(re.findall(re.compile(r"[0-9]"), tagg))

        # Calculate ratios
        ratio_spaces = num_spaces / total_chars
        ratio_alpha = num_alpha / total_chars
        ratio_numeric = num_numeric / total_chars
        ratio_parenthesis = num_parenthesis / total_chars
        ratio_slash = num_slash / total_chars
        ratio_plus = num_plus / total_chars
        ratio_point = num_point / total_chars
        ratio_comma = num_comma / total_chars
        ratio_semicolon = num_semicolon / total_chars

        # Calculate entropy
        ent = entropy(tagg)

        tagg_lower = tagg.lower()
        word_count = len(set([word for word in dictionary_words if word in tagg_lower]))
        
        enriched_rows.append([tagg] + [str(val) for val in [total_chars, num_spaces, num_parenthesis, num_slash, num_plus, num_point, num_comma, num_semicolon, num_alpha, num_numeric, ratio_spaces, ratio_alpha, ratio_numeric, ratio_parenthesis, ratio_slash, ratio_plus, ratio_point, ratio_comma, ratio_semicolon, ent, word_count]])
    return enriched_rows

if __name__ == "__main__":
    dictionary_words = create_dictionary_words()
    
    rows = (list(row) for row in pd.read_csv("tags.csv", encoding="ISO-8859-1").itertuples(index=False))
    row_chunks = [(chunk,) for chunk in chunked(rows, 100)]

    ray.shutdown()
    ray.init(include_dashboard=False)
    data = ["total_chars,num_spaces,num_parenthesis,num_slash,num_plus,num_point,num_comma,num_semicolon,num_alpha,num_numeric,ratio_spaces,ratio_alpha,ratio_numeric,ratio_parenthesis,ratio_slash,ratio_plus,ratio_point,ratio_comma,ratio_semicolon,ent,word_count"] + list(
        flatten(execute_with_ray(enrich_content_chunk, row_chunks, object_store={"dictionary_words": dictionary_words}))
        )

    with open("testPre2.csv", "w", encoding="utf-8") as enriched_csv:
        enriched_csv.write('\n'.join(data))
    ray.shutdown()


async def process_contents_chunk(urls, dictionary_words):
    loop = asyncio.get_event_loop()

    data = pd.read_csv("tags.csv", encoding="ISO-8859-1").itertuples(index=False)

    # Create a list of coroutine objects
    coroutines_contents = [enrich_content_chunk(data, dictionary_words)]
    
    # Run the coroutines concurrently using asyncio.gather
    results_content = await asyncio.gather(*coroutines_contents)    
    
    print(results_content)

    return results_content