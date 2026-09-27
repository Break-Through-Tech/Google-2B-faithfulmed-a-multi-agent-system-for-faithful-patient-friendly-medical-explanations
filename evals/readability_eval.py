"""
FaithfulMed — Raedability evaluation harness.

Provides:
  - is_refusal(text): whether or not the text is a refusal
  - evaluate_single(text): dictionary of Flesch-Kincaid grade, SMOG, jargon density, length, readability target success, and if it is a refusal
  - evaluate_multiple(texts): list of dictionaries of text, Flesch-Kincaid grades, SMOGs, jargon densities, lengths, readability target successes, and if it is a refusal
  - summarize_results(results): dictionary of mean Flesch-Kincaid grade, SMOG, jargon density, length, percentage of readability target success, and refusal rate
  - compare_scores(before, after): dictionary with both texts and difference between Flesch-Kincaid grade, SMOG, jargon density, and length for a text before and after
  - readability_target_rate(texts): percentage of readability target successes
  - find_failures(texts): list of dictionaries of text and Flesch-Kincaid grade
  - refusal_rate(texts): percentage of texts that are a refusal
"""

from pathlib import Path
import sys

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"
sys.path.append(str(NOTEBOOKS))

from eval_harness import (
    readability_scores,
    meets_readability_target
)


def is_refusal(text: str) -> bool:
    """
    Determines if the given text is a refusal.
    """
    phrases = [
        "i can't help",
        "i cannot help",
        "i can't provide",
        "i cannot provide",
        "i'm unable to",
        "i am unable to",
        "i can't answer",
        "i cannot answer"
    ]

    text_lower = text.lower()

    return any(phrase in text_lower for phrase in phrases)


def evaluate_single(text: str) -> dict:
    """
    Retrieves the core readability metrics for multiple pieces of text
    along with the readability target success, and whther it is a refusal or not.
    Creates a dictionary containing the readability score information, 
    readability target success, and whether it is a refusal.
    """
    scores = readability_scores(text)

    single = {
        **scores,
        "meets_target": meets_readability_target(text),
        "is_refusal": is_refusal(text)
    }

    return single


def evaluate_multiple(texts: list[str]) -> list[dict]:
    """
    Retrieves the core readability metrics for multiple pieces of text.
    Creates a list of dictionaries containing information about each 
    text, its readability score information, readability target
    success, and whether it is a refusal.
    """
    results = []

    for text in texts:
        res = evaluate_single(text)

        # Appending a dictionary containing the text, unpacked scores, and readability target success
        results.append({
            "text": text,
            **res,
        })

    return results


def summarize_results(results: list[dict]) -> list[dict]:
    """
    Summarizes the results from a list of evaluated texts.
    """
    n = len(results)

    # Creating a dictionary with all the total number of results, mean values, percentage of success, and refusal rate
    summary = {
        "num_results": n,
        "mean_fk_grade": sum(res["flesch_kincaid_grade"] for res in results) / n,
        "mean_smog_index": sum(res["smog_index"] for res in results) / n,
        "mean_jargon_density": sum(res["jargon_density"] for res in results) / n,
        "mean_word_count": sum(res["word_count"] for res in results) / n,
        "percent_successful": (sum(res["meets_target"] for res in results) / n) * 100,
        "percent_refusal": (sum(res["is_refusal"] for res in results) / n) * 100
    }

    return summary


def compare_scores(before: str, after: str) -> dict:
    """
    Compares readability scores for a piece of text before
    and after.
    """
    before_scores = readability_scores(before)
    after_scores = readability_scores(after)

    # Creating a dictionary with both texts and the difference in scores
    compare = {
        "before": before,
        "after": after,
        "fk_diff": after_scores["flesch_kincaid_grade"] - before_scores["flesch_kincaid_grade"],
        "smog_index_diff": after_scores["smog_index"] - before_scores["smog_index"],
        "jargon_density_diff": after_scores["jargon_density"] - before_scores["jargon_density"],
        "word_count_diff": after_scores["word_count"] - before_scores["word_count"]
    }

    return compare


def readability_target_rate(texts: list[str]) -> float:
    """
    Calculates and returns the percentage of texts that
    meet the readability target criteria.
    """
    return (sum(meets_readability_target(text) for text in texts) / len(texts)) * 100


def find_failures(texts: list[str]) -> list[dict]:
    """
    Retrieves the texts that did not meet the readability
    success criteria.
    """
    failures = []

    for text in texts:
        readability = meets_readability_target(text)

        if not readability:
            # Appending a dictionary containing the text and it's Flesch-Kincaid grade
            failures.append({
                "text": text,
                "score": readability_scores(text)["flesch_kincaid_grade"]
            })

    return failures


def refusal_rate(texts: list[str]) -> float:
    """
    Calculates and returns the percentage of texts that
    are a refusal
    """
    return (sum(is_refusal(text) for text in texts) / len(texts)) * 100

