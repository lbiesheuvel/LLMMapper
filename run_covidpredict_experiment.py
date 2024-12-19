from src.data.enums import VocabularyToken, PoolingMode
from src.sentence_transformers_model.train.MappingModel import MappingModel
from src.sentence_transformers_model.train.CustomTrainEvaluator import (
    CustomTrainEvaluator,
)
from src.data.sources.mapping_dataset import Parameter, Concept
import pandas as pd
from typing import List, Optional
import optuna
import torch


def run_covidpredict_experiment(
    # optional trial
    trial: Optional[optuna.Trial],
    model_name: str,
    forbidden_combinations: Optional[List[List[VocabularyToken]]],
    train_parameters: List[Parameter],
    test_parameters: List[Parameter],
    all_concepts: List[Concept],
    batch_size: int,
    include_ehr_token: bool,
    include_type_token: bool,
    include_vocab_token: bool,
    pooling_mode: PoolingMode,
    reduce_max_seq_length: bool,
    add_dense_layer: bool,
    dense_layer_out_features: int,
    warmup_steps_multiplier: float,
    num_epochs: int,
    learning_rate: float,
    num_eval_records: int,
    verbose: bool,
    include_evaluator: bool,
):

    model = MappingModel(
        model_name=model_name,
        include_query_text=True,
        include_ehr_token=include_ehr_token,
        include_type_token=include_type_token,
        include_vocab_token=include_vocab_token,
        pooling_mode=pooling_mode,
        reduce_max_seq_length=reduce_max_seq_length,
        add_dense_layer=add_dense_layer,
        dense_layer_out_features=dense_layer_out_features,
    )

    if include_evaluator:
        evaluator = CustomTrainEvaluator(
            trial=trial,
            test_parameters=test_parameters,
            all_concepts=all_concepts,
            include_query_text=True,
            include_ehr_token=include_ehr_token,
            include_type_token=include_type_token,
            include_vocab_token=include_vocab_token,
            show_progress_bar=verbose,
            print_results=verbose,
            num_test_records=num_eval_records,
        )

    model.train(
        lr=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        warmup_steps_multiplier=warmup_steps_multiplier,
        evaluator=evaluator if include_evaluator else None,
        train_parameters=train_parameters,
        all_concepts=all_concepts,
        forbidden_combinations=forbidden_combinations,
        show_progress_bar=verbose,
        verbose=verbose,
    )

    results = model.evaluate(
        test_parameters=test_parameters,
        all_concepts=all_concepts,
        show_progress_bar=verbose,
    )

    return {"model": model, "results": results}
