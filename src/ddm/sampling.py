"""Deterministic document splits and next-token windows for EXP-001."""
import hashlib
import random


def window_quotas(target_count, positions_per_window=32):
    if target_count <= 0 or positions_per_window <= 0 or target_count % (10 * positions_per_window):
        raise ValueError('Target count must allow exact whole-window 80/10/10 splits')
    unit = target_count // (10 * positions_per_window)
    return {'train':8*unit, 'validation':unit, 'test':unit}


def content_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def first_content_occurrence(text, seen_hashes):
    """Deduplicate exact text before splitting; preserve original corpus order."""
    digest = content_hash(text)
    if digest in seen_hashes:
        return False
    seen_hashes.add(digest)
    return True


def document_split(document_id: str, seed: int) -> str:
    bucket = int(hashlib.sha256(f'{seed}:{document_id}'.encode()).hexdigest(), 16) % 100
    return 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'


def sample_positions(document_id, window_id, seed, seq_len=1024, count=32, minimum=128):
    if not 0 <= minimum < seq_len - 1 or not 0 < count <= seq_len - 1 - minimum:
        raise ValueError('Invalid next-token position bounds')
    digest = hashlib.sha256(f'{seed}:{document_id}:{window_id}'.encode()).digest()
    return sorted(random.Random(int.from_bytes(digest, 'big')).sample(range(minimum, seq_len - 1), count))


def first_window(tokens, seq_len=1024):
    """Smoke policy: one non-overlapping full window per eligible document."""
    return tokens[:seq_len] if len(tokens) >= seq_len else None
