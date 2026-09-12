"""
src/data_cleaning.py
--------------------
Text cleaning, normalization, and quality evaluation utilities for Twitter
Customer Support data, with specific tuning for AppleSupport interactions.

Guiding Principles:
- Preserve authentic, noisy customer language (slang, typos, casing, emotions).
- Remove structural Twitter noise (leading/trailing @mentions, HTML artifacts like &amp;).
- Filter degenerate/empty messages without discarding real short inquiries.
- Provide tagging for DM deflection and diagnostic resolution steps.
"""

import html
import re

# Regex patterns
RE_MENTIONS = re.compile(r"@[A-Za-z0-9_]+")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_WHITESPACE = re.compile(r"[ \t]+")
RE_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
RE_AGENT_SIGNATURE = re.compile(r"\^[A-Z]{2,4}$|/[A-Z]{2,4}$")

DM_PATTERNS = [
    r"\bdm\b",
    r"\bdms\b",
    r"\bdirect message\b",
    r"\bprivate message\b",
    r"\bpm\b",
    r"send us a message",
    r"reach out via dm",
    r"shoot us a dm",
    r"send a dm",
    r"send me a dm",
    r"join us in a dm",
    r"click the link to dm",
    r"click here to dm",
    r"select the following link to join us in a dm"
]
RE_DM = re.compile("|".join(DM_PATTERNS), re.IGNORECASE)

RESOLUTION_PATTERNS = [
    r"\btry\b",
    r"\brestart\b",
    r"\breinstall\b",
    r"\bsettings\b",
    r"\bsteps\b",
    r"\bupdate\b",
    r"\breset\b",
    r"\barticle\b",
    r"\bguide\b",
    r"\bcheck if\b",
    r"\bmake sure\b",
    r"\bturn off\b",
    r"\bturn on\b",
    r"\bforce close\b",
    r"\bgo to\b",
    r"\btap on\b",
    r"\bclick on\b",
    r"\bversion\b",
    r"\bbackup\b",
    r"\brestore\b",
    r"\biCloud\b",
    r"\bavailable space\b",
    r"\bstorage\b"
]
RE_RESOLUTION = re.compile("|".join(RESOLUTION_PATTERNS), re.IGNORECASE)


def clean_tweet_text(
    text: str,
    strip_mentions: bool = True,
    preserve_urls: bool = True
) -> str:
    """
    Cleans raw tweet text while preserving authentic customer phrasing.

    Args:
        text: Raw tweet text string.
        strip_mentions: Whether to remove @mentions.
        preserve_urls: Whether to keep URLs (useful for support links).

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    # Unescape HTML entities (e.g. &amp; -> &, &lt; -> <)
    cleaned = html.unescape(text)

    # Normalize weird unicode spaces and characters
    cleaned = cleaned.replace("\u200b", "").replace("\ufeff", "")

    # Strip @mentions if requested
    if strip_mentions:
        cleaned = RE_MENTIONS.sub("", cleaned)

    # Strip URLs if requested
    if not preserve_urls:
        cleaned = RE_URL.sub("", cleaned)

    # Normalize excessive newlines and whitespace
    cleaned = RE_MULTIPLE_NEWLINES.sub("\n\n", cleaned)
    cleaned = RE_WHITESPACE.sub(" ", cleaned)

    return cleaned.strip()


def is_usable_customer_message(text: str, min_chars: int = 10, min_words: int = 2) -> bool:
    """
    Determines if a customer message contains usable, substantive content.

    Discards:
    - Empty strings
    - Pure @mentions or pure URLs
    - Messages with fewer than min_chars or min_words
    """
    if not isinstance(text, str):
        return False

    # Check content stripped of mentions AND urls
    stripped = clean_tweet_text(text, strip_mentions=True, preserve_urls=False)
    stripped = re.sub(r"[^\w\s]", "", stripped).strip()

    if len(stripped) < min_chars:
        return False

    words = stripped.split()
    return len(words) >= min_words


def is_usable_support_reply(text: str, min_chars: int = 15, min_words: int = 3) -> bool:
    """
    Determines if a support reply has substantive content.
    """
    if not isinstance(text, str):
        return False

    cleaned = clean_tweet_text(text, strip_mentions=True, preserve_urls=True)
    if len(cleaned) < min_chars:
        return False

    words = cleaned.split()
    return len(words) >= min_words


def detect_dm_deflection(text: str) -> bool:
    """
    Returns True if the text indicates a deflection to private messaging / DM.
    """
    if not isinstance(text, str):
        return False
    return bool(RE_DM.search(text))


def detect_troubleshooting(text: str) -> bool:
    """
    Returns True if the text contains troubleshooting instructions or resolution steps.
    """
    if not isinstance(text, str):
        return False
    return bool(RE_RESOLUTION.search(text))
