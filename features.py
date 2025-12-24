import re
import math
import socket
import requests
import tldextract
from urllib.parse import urlparse
from typing import Dict, Tuple

SUSPICIOUS_KEYWORDS = [
    'login','verify','account','update','secure','confirm','bank','payment','invoice',
    'free','prize','win','offer','gift','limited','urgent','support','security','unlock',
    'apple','paypal','amazon','microsoft','google','wallet'
]

SUSPICIOUS_TLDS = {
    'tk','ml','ga','cf','gq','xyz','top','work','click','country','stream','download','zip'
}

HEURISTIC_WEIGHTS = {
    'has_ip': 14,
    'url_length': 12,
    'entropy': 10,
    'num_subdomains': 8,
    'num_params': 8,
    'has_at': 10,
    'has_dash': 6,
    'has_multiple_slashes': 6,
    'suspicious_tld': 8,
    'suspicious_keywords': 12,
    'redirects': 6,
    'http_only': 8,
}

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter
    c = Counter(s)
    n = len(s)
    return -sum((cnt/n) * math.log2(cnt/n) for cnt in c.values())

def is_ip(host: str) -> bool:
    try:
        socket.inet_aton(host)  # IPv4
        return True
    except OSError:
        pass
    return bool(re.match(r'^[0-9a-fA-F:]+$', host))  # very loose IPv6

def extract_url_features(url: str) -> Dict[str, float]:
    if not (url.startswith('http://') or url.startswith('https://')):
        url = 'http://' + url
    parsed = urlparse(url)
    ext = tldextract.extract(url)
    host = parsed.hostname or ''
    path = parsed.path or ''
    query = parsed.query or ''

    url_no_scheme = url.split('://', 1)[-1]
    feats = {}
    feats['url_length'] = len(url)
    feats['host_length'] = len(host)
    feats['path_length'] = len(path)
    feats['num_params'] = query.count('=')
    feats['num_subdirs'] = path.count('/')
    feats['num_subdomains'] = len([p for p in ext.subdomain.split('.') if p]) if ext.subdomain else 0
    feats['has_at'] = int('@' in url_no_scheme)
    feats['has_dash'] = int('-' in host)
    feats['has_multiple_slashes'] = int(url_no_scheme.count('//') > 0)
    feats['suspicious_tld'] = int((ext.suffix or '').lower() in SUSPICIOUS_TLDS)
    feats['has_ip'] = int(is_ip(host))
    feats['entropy'] = shannon_entropy(host + path)
    lower_all = (host + ' ' + path + ' ' + query).lower()
    feats['suspicious_keywords'] = sum(1 for k in SUSPICIOUS_KEYWORDS if k in lower_all)
    feats['http_only'] = int(parsed.scheme.lower() == 'http')
    return feats

def soft_http_checks(url: str, timeout: float = 4.0) -> Dict[str, float]:
    """Lightweight network checks (optional, short timeouts)."""
    info = {'reachable': 0, 'status_code': 0, 'redirects': 0}
    try:
        r = requests.get(url if url.startswith('http') else ('http://' + url), timeout=timeout, allow_redirects=True)
        info['reachable'] = 1
        info['status_code'] = r.status_code
        info['redirects'] = len(r.history)
    except Exception:
        pass
    return info

def heuristic_score(url: str) -> Tuple[int, str, Dict[str, float]]:
    f = extract_url_features(url)
    net = soft_http_checks(url)
    score = 0
    w = HEURISTIC_WEIGHTS
    score += w['has_ip'] * f['has_ip']
    score += w['url_length'] * (1 if f['url_length'] > 75 else 0)
    score += w['entropy'] * (1 if f['entropy'] > 3.5 else 0)
    score += w['num_subdomains'] * (1 if f['num_subdomains'] >= 3 else 0)
    score += w['num_params'] * (1 if f['num_params'] >= 3 else 0)
    score += w['has_at'] * f['has_at']
    score += w['has_dash'] * f['has_dash']
    score += w['has_multiple_slashes'] * f['has_multiple_slashes']
    score += w['suspicious_tld'] * f['suspicious_tld']
    score += w['suspicious_keywords'] * (1 if f['suspicious_keywords'] >= 2 else 0)
    score += w['redirects'] * (1 if net['redirects'] >= 3 else 0)
    score += w['http_only'] * f['http_only']

    score = max(0, min(100, score))
    if score >= 65:
        label = 'Fraudulent'
    elif score >= 35:
        label = 'Suspicious'
    else:
        label = 'Safe'
    f.update(net)
    f['risk_score'] = score
    return score, label, f
