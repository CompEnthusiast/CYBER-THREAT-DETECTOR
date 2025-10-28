# scripts/feature_extractor.py
import re
from urllib.parse import urlparse

def extract_features_from_url(url: str) -> dict:
    """
    Extracts numerical features from a given URL for phishing detection.
    Returns a dictionary suitable for ML model input.
    """

    # Safeguard for None or invalid URLs
    if not isinstance(url, str) or not url.strip():
        return {f: 0 for f in FEATURES_LIST}

    # Parse URL
    parsed = urlparse(url)
    hostname = parsed.netloc or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # Basic length features
    url_length = len(url)
    hostname_length = len(hostname)
    path_length = len(path)
    query_length = len(query)

    # Count-based features
    num_dots = url.count(".")
    num_hyphens = url.count("-")
    num_digits = sum(c.isdigit() for c in url)
    num_special_chars = sum(c in "@!#$%^&*()+=[]{}|\\;:'\",<>?/" for c in url)

    # Boolean features (as 0/1)
    has_https = int(url.startswith("https"))
    has_ip = int(bool(re.search(r'(\d{1,3}\.){3}\d{1,3}', hostname)))
    has_at_symbol = int("@" in url)
    has_double_slash = int(url.count("//") > 1)
    has_suspicious_tld = int(any(tld in hostname for tld in [".xyz", ".top", ".info", ".club", ".online", ".site", ".cc", ".biz"]))

    # Keyword-based features
    keywords = ['login', 'secure', 'bank', 'verify', 'update', 'free', 'click', 'confirm', 'account', 'signin']
    keyword_count = sum(kw in url.lower() for kw in keywords)

    # Domain entropy (rough estimate)
    entropy = 0
    for char in set(hostname):
        p = hostname.count(char) / len(hostname)
        entropy -= p * (p and __import__('math').log2(p))

    # Combine all features
    features = {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "query_length": query_length,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_digits": num_digits,
        "num_special_chars": num_special_chars,
        "has_https": has_https,
        "has_ip": has_ip,
        "has_at_symbol": has_at_symbol,
        "has_double_slash": has_double_slash,
        "has_suspicious_tld": has_suspicious_tld,
        "keyword_count": keyword_count,
        "entropy": round(entropy, 4),
    }

    return features


# Helps model builder know the expected feature keys
FEATURES_LIST = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "num_special_chars",
    "has_https",
    "has_ip",
    "has_at_symbol",
    "has_double_slash",
    "has_suspicious_tld",
    "keyword_count",
    "entropy",
]

# Quick test
if __name__ == "__main__":
    test_url = "https://secure-login.bank-update.xyz/verify?user=123"
    print(extract_features_from_url(test_url))
