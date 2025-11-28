import re
from difflib import SequenceMatcher

# accuracy.py
def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            insert_cost = prev[j] + 1
            delete_cost = cur[j - 1] + 1
            replace_cost = prev[j - 1] + (0 if ca == cb else 1)
            cur[j] = min(insert_cost, delete_cost, replace_cost)
        prev = cur
    return prev[-1]


def normalize_text(text):
    """Remove extra whitespace and normalize text for comparison"""
    if not text:
        return ""
    # Collapse multiple spaces into single space
    text = re.sub(r'\s+', ' ', text.strip())
    return text.lower()


def normalize_whitespace_only(text):
    """Normalize whitespace but preserve case for display"""
    if not text:
        return ""
    # Collapse multiple spaces/tabs/newlines into single space
    return re.sub(r'\s+', ' ', text.strip())


def extract_redacted_entities(text):
    """Extract all redaction markers from text"""
    if not text:
        return []

    patterns = [
        r'\[ID\]', r'\[CC\]', r'\[SSN\]', r'\[NAME\]', r'\[EMAIL\]',
        r'\[PHONE\]', r'\[ADDRESS\]', r'\[ZIP\]', r'\[API_KEY\]',
        r'\[FILE\]', r'\[IP\]', r'\[DATETIME\]', r'\[PERCENT\]',
        r'\[ROUTING\]', r'\[ACCOUNT\]', r'\[PASSPORT\]', r'\[DL\]',
        r'\[VIN\]', r'\[MAC\]', r'\[MED_ID\]', r'\[EMP_ID\]', r'\[COORDS\]',
        r'<PERSON>', r'<LOCATION>', r'<EMAIL_ADDRESS>', r'<PHONE_NUMBER>',
        r'<DATE>', r'<TIME>', r'<CREDIT_CARD>', r'<IP_ADDRESS>'
    ]

    found = []
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            found.append({
                'type': match.group(0),
                'start': match.start(),
                'end': match.end()
            })

    return sorted(found, key=lambda x: x['start'])


def calculate_precision_recall(predicted, ground_truth):
    """Calculate precision and recall for entity detection"""
    # Normalize whitespace before entity extraction
    pred_normalized = normalize_whitespace_only(predicted)
    gt_normalized = normalize_whitespace_only(ground_truth)

    if not pred_normalized and not gt_normalized:
        return 1.0, 1.0, 1.0

    if not pred_normalized:
        return 0.0, 0.0, 0.0

    if not gt_normalized:
        return 0.0, 1.0, 0.0

    pred_entities = extract_redacted_entities(pred_normalized)
    true_entities = extract_redacted_entities(gt_normalized)

    if not pred_entities and not true_entities:
        return 1.0, 1.0, 1.0

    if not pred_entities:
        return 0.0, 0.0, 0.0

    if not true_entities:
        return 0.0, 1.0, 0.0

    true_positives = 0
    matched_true = set()

    for pred in pred_entities:
        for idx, true in enumerate(true_entities):
            if idx in matched_true:
                continue

            overlap_start = max(pred['start'], true['start'])
            overlap_end = min(pred['end'], true['end'])

            if overlap_start < overlap_end and pred['type'] == true['type']:
                true_positives += 1
                matched_true.add(idx)
                break

    false_positives = len(pred_entities) - true_positives
    false_negatives = len(true_entities) - true_positives

    precision = true_positives / len(pred_entities) if pred_entities else 0.0
    recall = true_positives / len(true_entities) if true_entities else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1_score


def calculate_text_similarity(text1, text2):
    """Calculate similarity ratio between two texts"""
    if not text1 and not text2:
        return 1.0

    if not text1 or not text2:
        return 0.0

    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)

    return SequenceMatcher(None, norm1, norm2).ratio()


def calculate_match_accuracy(predicted, ground_truth):
    """Calculate how well the redacted output matches expected output"""
    if not predicted and not ground_truth:
        return 100.0

    # Normalize whitespace before comparison
    pred_normalized = normalize_whitespace_only(predicted)
    gt_normalized = normalize_whitespace_only(ground_truth)

    similarity = calculate_text_similarity(pred_normalized, gt_normalized)
    return round(similarity * 100, 2)


def calculate_content_preservation(original, redacted):
    """Calculate how much original content was preserved vs removed"""
    if not original:
        return 0.0

    if not redacted:
        return 0.0

    # Normalize for fair comparison
    orig_normalized = normalize_text(original)
    redacted_normalized = normalize_text(redacted)

    orig_tokens = set(orig_normalized.split())
    redacted_tokens = set(redacted_normalized.split())

    entity_markers = {
        '[id]', '[cc]', '[ssn]', '[name]', '[email]', '[phone]',
        '[address]', '[zip]', '[api_key]', '[file]', '[ip]',
        '[datetime]', '[percent]', '[routing]', '[account]',
        '[passport]', '[dl]', '[vin]', '[mac]', '[med_id]',
        '[emp_id]', '[coords]', '<person>', '<location>',
        '<email_address>', '<phone_number>', '<date>', '<time>',
        '<credit_card>', '<ip_address>'
    }

    redacted_tokens = redacted_tokens - entity_markers

    if not orig_tokens:
        return 100.0

    preserved = len(redacted_tokens.intersection(orig_tokens))
    preservation_rate = (preserved / len(orig_tokens)) * 100

    return round(preservation_rate, 2)


def calculate_entity_metrics(predicted, ground_truth):
    """Calculate detailed entity-level metrics"""
    precision, recall, f1 = calculate_precision_recall(predicted, ground_truth)

    return {
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1_score': round(f1 * 100, 2)
    }


def calculate_accuracy(original, redacted, ground_truth):
    """
    Main accuracy calculation function with whitespace normalization

    Args:
        original: Original unredacted text
        redacted: Your system's redacted output
        ground_truth: Expected redacted output

    Returns:
        dict: Comprehensive accuracy metrics
    """

    match_acc = calculate_match_accuracy(redacted, ground_truth)
    content_pres = calculate_content_preservation(original, redacted)
    entity_metrics = calculate_entity_metrics(redacted, ground_truth)

    overall_score = (
        match_acc * 0.4 +
        entity_metrics['f1_score'] * 0.4 +
        content_pres * 0.2
    )

    return {
        'Match Accuracy': match_acc,
        'Content Preservation': content_pres,
        'Precision': entity_metrics['precision'],
        'Recall': entity_metrics['recall'],
        'F1 Score': entity_metrics['f1_score'],
        'Overall Score': round(overall_score, 2)
    }


def generate_detailed_report(original, redacted, ground_truth):
    """Generate a detailed accuracy report with entity breakdown"""

    results = calculate_accuracy(original, redacted, ground_truth)

    pred_entities = extract_redacted_entities(redacted)
    true_entities = extract_redacted_entities(ground_truth)

    entity_counts = {}
    for entity in pred_entities:
        entity_type = entity['type']
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

    report = {
        'metrics': results,
        'detected_entities': len(pred_entities),
        'expected_entities': len(true_entities),
        'entity_breakdown': entity_counts,
        'text_length': {
            'original': len(original),
            'redacted': len(redacted),
            'reduction': len(original) - len(redacted)
        }
    }

    return report
