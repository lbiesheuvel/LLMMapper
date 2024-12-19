from typing import List, Tuple
from sentence_transformers import InputExample
from src.data.enums import VocabularyToken


def get_input_examples(
    parameter_queries: List[str],
    concept_queries: List[str],
    vocabulary_tokens: List[VocabularyToken],
) -> Tuple[List[InputExample], List[VocabularyToken]]:
    return [
        InputExample(texts=[parameter_query, concept_query], label=1.0)
        for (parameter_query, concept_query) in zip(parameter_queries, concept_queries)
    ], vocabulary_tokens
