import hashlib

def generate_hash(doc_id: str, question: str) -> str:
    """Generate an MD5 hash based on document ID and question."""
    hash_str = f"{doc_id}_{question}"
    return hashlib.md5(hash_str.encode()).hexdigest() 