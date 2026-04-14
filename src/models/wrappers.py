from __future__ import annotations


def extract_logits(output):
    if isinstance(output, dict):
        return output["out"]
    return output

