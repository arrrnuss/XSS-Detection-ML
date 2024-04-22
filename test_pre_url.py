import pandas as pd
import ray
from more_itertools import chunked, flatten
from urllib.parse import urlparse, quote, unquote, parse_qs
import math
import string
from math import log2
from collections import Counter
import asyncio
import re

from parallel_compute import execute_with_ray

NUMBERS = set("0123456789")
SPEC_CHARS = set(["+", '"', "*", "#", "%", "&", "(", ")", "=", "?", "^", "-", ".", "!", "~", "_", ">", "<"])
ALL_CHARS = set(string.printable) - set('\x0b\f\r')

def create_dictionary_words() -> set[str]:
    df = pd.read_csv('listofattack.csv')
    dictionary_words: set[str] = set(df['Word'].apply(str))
    return dictionary_words

async def enrich_row_chunk(url, dictionary_words):
    print("Processing URL:", url)
    enriched_rows = []  # Initialize an empty list for the current URL

    spec_chars = 0
    depth = 0
    numericals_count = 0
    word_count = 0
    url = str(url[0])

    # Protocol feature
    if urlparse(url).scheme == "https":
        protocol = 0
    elif urlparse(url).scheme == "http":
        protocol = 1
    else:
        protocol = 2

    print("URL:", url)
    print("Extracted Scheme:", urlparse(url).scheme)
    print("Type of 'urls':", type(url))
    print("Protocol:", protocol)


    # URL Path Length feature
    url_path_len = len(urlparse(url).path)

    # Port feature
    port = urlparse(url)
    if port.scheme == "https":
        if port.port is None or port.port == 443:
            port = 443
        else:
            port = port.port
    elif port.scheme == "http":
        if port.port is None or port.port == 80:
            port = 80
        else:
            port = port.port
    else:
        port = 0

    # URL Entropy feature   
    # Count the number of occurrences of each character in the URL
    char_counts = Counter(url)

    # Calculate the frequency of each character in the URL
    url_len = len(url)
    char_freqs = [count/url_len for count in char_counts.values()]

    # Calculate the Shannon entropy of the URL
    entropy = -sum([freq*log2(freq) for freq in char_freqs])
    print("entropy",entropy)

    # URL Encoded feature
    url_encoded = urlparse(url)
    unquoted_url = unquote(url)
    url_encoded = int(url_encoded.geturl() != unquoted_url or unquoted_url != url)

    print("url encoded",url_encoded)

    parsed_url = urlparse(url)
    if parsed_url.hostname:
        tld = parsed_url.hostname.split('.')[-1]
        if tld in ('com', 'edu', 'gov', 'mil', 'int'):
            tld = 1
        elif tld in ('co', 'th', 'in', 'io', 'me', 'ly', 'fm'):
            tld = 2
        elif tld in ('org'):
            tld = 3
        elif tld in ('net'):
            tld = 4
        else:
            tld = 0
    else:
        tld = 0  # Default value for invalid URLs

    url_lower = url.lower()
    word_count = len(set([word for word in dictionary_words if word in url_lower]))
    for c in url:
        if c in SPEC_CHARS:
            spec_chars += 1
        elif c in ["/"]:
            depth += 1
        elif c in NUMBERS:
            numericals_count += 1


    parsed_url = urlparse(url)
    domain_L = len(parsed_url.netloc) # Domain Length
    number_subdomains = parsed_url.netloc.count('.') - 1
    number_dots = parsed_url.netloc.count('.')
    number_hyphens = parsed_url.netloc.count('-')
    number_QueryP = len(parse_qs(parsed_url.query))
    domain_entropy = calculate_domain_entropy(parsed_url.netloc)
    r_sp = sum(1 for char in url if not char.isalnum()) / len(url)
    r_digits = sum(1 for char in url if char.isdigit()) / len(url)
    r_slashes = parsed_url.path.count('/') / len(url)
    pre_redirects = 1 if re.search(r'(https?|www)', parsed_url.path) else 0
    r_letters = sum(1 for char in url if char.isalpha()) / len(url)


    enriched_row = f"{protocol},{len(url)},{url_path_len},{port},{tld},{entropy:.2f},{spec_chars},{depth},{numericals_count},{word_count},{url_encoded},{domain_L},{number_subdomains},{number_dots},{number_hyphens},{number_QueryP},{domain_entropy},{r_sp},{r_digits},{r_slashes},{pre_redirects},{r_letters}"

    # enriched_row = f"{protocol},{len(url)},{url_path_len},{port},{tld},{entropy:.2f},{spec_chars},{depth},{numericals_count},{word_count},{url_encoded}"
    enriched_rows.append(enriched_row)  # Append the individual enriched row for this URL
    print("Pre URL: ",enriched_rows)

    return enriched_rows

def calculate_domain_entropy(domain):
    char_count = {}
    for char in domain:
        char_count[char] = char_count.get(char, 0) + 1

    entropy = 0.0
    domain_length = len(domain)
    for char in char_count:
        char_probability = char_count[char] / domain_length
        entropy -= char_probability * math.log2(char_probability)
    
    return entropy

if __name__ == "__main__":
    dictionary_words = create_dictionary_words()

    rows = (list(row) for row in pd.read_csv("test2.csv").itertuples(index=False))
    row_chunks = [(chunk,) for chunk in chunked(rows, 100)]

    ray.shutdown()
    ray.init(include_dashboard=False)
    data = ["protocol,len,url_path_len,port,domain,entropy,spec_chars,depth,numericals_count,word_count,url_encoded,label"] + list(
        flatten(execute_with_ray(enrich_row_chunk, row_chunks, object_store={"dictionary_words": dictionary_words}))
    )
    with open("Content_enriched_data7.csv", "w") as enriched_csv:
        enriched_csv.write('\n'.join(data))
    ray.shutdown()


async def process_urls_chunk(urls, dictionary_words):
    loop = asyncio.get_event_loop()

    # Create a list of coroutine objects
    coroutines = [enrich_row_chunk(urls, dictionary_words)]
    # Run the coroutines concurrently using asyncio.gather
    results = await asyncio.gather(*coroutines)

    return results