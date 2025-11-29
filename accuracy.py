import re
from difflib import SequenceMatcher

def tokenize(text):
    """
    Robust tokenization that splits punctuation.
    Ensures "IP." and "IP ." are treated as the same sequence of tokens.
    """
    if not text:
        return []
    # Add spaces around punctuation to isolate them as tokens
    text = re.sub(r'([.,!?;:])', r' \1 ', text)
    # Split by whitespace
    return text.lower().split()

def get_diff_indices(seq1, seq2):
    """
    Compare seq1 and seq2 using Difflib.
    Returns a set of indices in seq1 that were changed (replaced/deleted) in seq2.
    """
    matcher = SequenceMatcher(None, seq1, seq2)
    changed_indices = set()
    
    # 'replace' and 'delete' indicate the token in seq1 is gone in seq2
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ('replace', 'delete'):
            for k in range(i1, i2):
                changed_indices.add(k)
                
    return changed_indices

def calculate_accuracy(original, redacted, ground_truth):
    """
    Robust accuracy calculation using Sequence Alignment.
    
    Returns:
        total_pii (int): Count of tokens that SHOULD be redacted.
        correct (int): Count of PII tokens successfully redacted.
        missed (int): Count of PII tokens that leaked through.
        precision (float): Quality (0.0 - 1.0)
        recall (float): Accuracy/Coverage (0.0 - 1.0)
        f1 (float): Harmonic mean (0.0 - 1.0)
    """
    
    # 1. Tokenize inputs robustly
    orig_tokens = tokenize(original)
    red_tokens = tokenize(redacted)
    gt_tokens = tokenize(ground_truth)
    
    if not orig_tokens:
        return 0, 0, 0, 0.0, 0.0, 0.0

    # 2. Identify Indices using Diff Algorithm (Shift-Resistant)
    
    # PII = Indices in Original that differ in Ground Truth
    pii_indices = get_diff_indices(orig_tokens, gt_tokens)
    
    # Redacted = Indices in Original that differ in System Output
    redacted_indices = get_diff_indices(orig_tokens, red_tokens)
    
    # 3. Calculate Intersection Stats
    
    # True Positive: It was PII, and we Redacted it.
    true_positives = pii_indices.intersection(redacted_indices)
    correct_count = len(true_positives)
    
    # False Negative (Missed): It was PII, but we did NOT redact it.
    missed_indices = pii_indices - redacted_indices
    missed_count = len(missed_indices)
    
    # False Positive (Over-redaction): It was NOT PII, but we Redacted it.
    over_redacted_indices = redacted_indices - pii_indices
    over_redacted_count = len(over_redacted_indices)
    
    total_pii = len(pii_indices)
    
    # 4. Metrics Calculation
    
    # Precision: Out of everything we redacted, how much was actually PII?
    precision = 1.0
    if (correct_count + over_redacted_count) > 0:
        precision = correct_count / (correct_count + over_redacted_count)
        
    # Recall (Accuracy): Out of actual PII, how much did we catch?
    recall = 1.0
    if total_pii > 0:
        recall = correct_count / total_pii
    elif over_redacted_count > 0:
        # If there was no PII to find, but we blindly redacted things, accuracy drops
        recall = 0.0
        
    f1 = 0.0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return total_pii, correct_count, missed_count, precision, recall, f1