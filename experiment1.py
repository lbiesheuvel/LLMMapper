# Experiment that takes 3 the data from the 3 hospitals from T. Dam paper and trains a model with it. Then, it evaluates performance on the remainder of hospitals.
MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 32
SEED = 42

from src.data.sources.covidpredict_mappings import CovidPredictMappings
from src.data.enums import VocabularyToken, PoolingMode
from run_covidpredict_experiment import (
    run_covidpredict_experiment,
)


def run_experiment_one():
    covidpredict_dataset = CovidPredictMappings(
        extract_after_last_semicolon=True, only_active_concepts=True, limited_mode=False
    )
    covidpredict_dataset.filter(only_mapped=True)

    train_idx, test_idx = covidpredict_dataset.get_tariq_splits()

    parameters_train = covidpredict_dataset.get_parameters(indices=train_idx)
    parameters_test = covidpredict_dataset.get_parameters(indices=test_idx)
    all_concepts = covidpredict_dataset.get_concepts()

    forbidden_combinations = [
        [VocabularyToken.ICUNITY, VocabularyToken.ICUDATA, VocabularyToken.LOINC],
        [VocabularyToken.RXNORM, VocabularyToken.ATC],
    ]

    output = run_covidpredict_experiment(
        trial=None,
        model_name=MODEL_NAME,
        train_parameters=parameters_train,
        test_parameters=parameters_test,
        all_concepts=all_concepts,
        forbidden_combinations=forbidden_combinations,
        batch_size=BATCH_SIZE,
        include_ehr_token=True,
        include_type_token=True,
        include_vocab_token=True,
        pooling_mode=PoolingMode.MEAN,
        reduce_max_seq_length=False,
        add_dense_layer=False,
        dense_layer_out_features=None,
        warmup_steps_multiplier=0.1,
        num_epochs=20,
        learning_rate=2e-5,
        num_eval_records=5000,
        verbose=True,
        include_evaluator=True,
    )

    def print_evaluation_results(results):
        accuracies = results["accuracies"]
        # metrics = results["metrics"]
        incorrect_predictions = results["incorrect_predictions"]

        print("Accuracies:")
        print(f"  Top-1 Accuracy: {accuracies['top_1_accuracy']:.3f}")
        print(f"  Top-5 Accuracy: {accuracies['top_5_accuracy']:.3f}")
        print(f"  Top-10 Accuracy: {accuracies['top_10_accuracy']:.3f}")
        print(f"  Top-100 Accuracy: {accuracies['top_100_accuracy']:.3f}")

    print_evaluation_results(output["results"])


if __name__ == "__main__":
    run_experiment_one()
