def to_bytes(s):
    if isinstance(s, bytes):
        return s
    return str(s).encode("utf-8", errors="ignore")


def to_unicode(s):
    if isinstance(s, str):
        return s
    return bytes(s).decode("utf-8", errors="ignore")
