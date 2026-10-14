"""Explicitly allowlisted experimental learning extension point for FINAL26.

The autonomous source-evolution mechanism may modify this module only after hard
sandbox validation. Security, reward, objective, browser, configuration, and
orchestration code remains immutable to model-generated source edits.
"""


def transform_training_example(example):
    """Return the example unchanged until an experiment is intentionally installed."""
    return example
