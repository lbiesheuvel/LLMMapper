EXPERIMENT_NAME = "Covidpredict_many_hospitals_november_2024-6"
N_TRIALS = 50
MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 32
SEED = 42

from src.data.sources.covidpredict_mappings import CovidPredictMappings
from src.data.enums import VocabularyToken, PoolingMode
from run_covidpredict_experiment import (
    run_covidpredict_experiment,
)
import optuna
import os


def run_experiment_two():
    covidpredict_dataset = CovidPredictMappings(
        extract_after_last_semicolon=True, only_active_concepts=True, limited_mode=False
    )
    covidpredict_dataset.filter(only_mapped=True)

    test_hospitals_idx, full_train_hospitals_idx = covidpredict_dataset.get_splits(
        n_hospitals_first_split=3,
        n_hospitals_second_split=99,
        first_split_ehr="MIX",
        second_split_ehr="MIX",
        seed=SEED,
    )
    val_hospitals_idx, train_hospitals_idx = covidpredict_dataset.get_splits(
        starting_idx=full_train_hospitals_idx,
        n_hospitals_first_split=3,
        n_hospitals_second_split=99,
        first_split_ehr="MIX",
        second_split_ehr="MIX",
        seed=SEED,
    )

    forbidden_combinations = [
        [VocabularyToken.ICUNITY, VocabularyToken.ICUDATA, VocabularyToken.LOINC],
        [VocabularyToken.RXNORM, VocabularyToken.ATC],
    ]

    def objective(trial):
        include_ehr_token = trial.suggest_categorical(
            "include_ehr_token", [True, False]
        )
        include_type_token = trial.suggest_categorical(
            "include_type_token", [True, False]
        )
        include_vocab_token = trial.suggest_categorical(
            "include_vocab_token", [True, False]
        )
        pooling_mode = trial.suggest_categorical("pooling_mode", ["CLS", "MEAN"])
        add_dense_layer = trial.suggest_categorical("add_dense_layer", [True, False])
        warmup_steps_multiplier = trial.suggest_float(
            "warmup_steps_multiplier", 0.05, 0.2, log=True
        )
        num_epochs = 20  # trial.suggest_int("num_epochs", 10, 20)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)

        output = run_covidpredict_experiment(
            trial=trial,
            model_name=MODEL_NAME,
            forbidden_combinations=forbidden_combinations,
            train_parameters=covidpredict_dataset.get_parameters(
                indices=train_hospitals_idx
            ),
            test_parameters=covidpredict_dataset.get_parameters(
                indices=val_hospitals_idx
            ),
            all_concepts=covidpredict_dataset.get_concepts(),
            batch_size=BATCH_SIZE,
            include_ehr_token=include_ehr_token,
            include_type_token=include_type_token,
            include_vocab_token=include_vocab_token,
            pooling_mode=(
                PoolingMode.MEAN if pooling_mode == "MEAN" else PoolingMode.CLS
            ),
            reduce_max_seq_length=False,
            add_dense_layer=add_dense_layer,
            dense_layer_out_features=768,
            warmup_steps_multiplier=warmup_steps_multiplier,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            num_eval_records=5000,
            verbose=True,
            include_evaluator=True,
        )

        results = output["results"]
        top_1_accuracy = results["accuracies"]["top_1_accuracy"]
        top_5_accuracy = results["accuracies"]["top_5_accuracy"]
        top_10_accuracy = results["accuracies"]["top_10_accuracy"]
        top_100_accuracy = results["accuracies"]["top_100_accuracy"]

        # save to .txt file (append, create if it doesnt exist)
        # create dir if it doesnt exist
        os.makedirs("src/results", exist_ok=True)
        with open(f"src/results/{EXPERIMENT_NAME}.txt", "a") as f:
            trial_number = len(study.trials) - 1
            f.write(
                f"Trial {trial_number}: Top-1 Accuracy: {top_1_accuracy:.3f}, Top-5 Accuracy: {top_5_accuracy:.3f}, Top-10 Accuracy: {top_10_accuracy:.3f}, Top-100 Accuracy: {top_100_accuracy:.3f}\n"
            )

        print(
            f"Top-1 Accuracy: {top_1_accuracy:.3f}, Top-5 Accuracy: {top_5_accuracy:.3f}, Top-10 Accuracy: {top_10_accuracy:.3f}, Top-100 Accuracy: {top_100_accuracy:.3f}"
        )

        return top_5_accuracy

    database_url = # LEFT OUT FOR SECURITY REASONS
    pruner = optuna.pruners.HyperbandPruner(min_resource=3)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{EXPERIMENT_NAME}",
        storage=database_url,
        load_if_exists=True,
        pruner=pruner,
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params

    # Train model with full train set and best params

    output = run_covidpredict_experiment(
        trial=None,
        model_name=MODEL_NAME,
        forbidden_combinations=forbidden_combinations,
        train_parameters=covidpredict_dataset.get_parameters(
            indices=full_train_hospitals_idx
        ),
        test_parameters=covidpredict_dataset.get_parameters(indices=test_hospitals_idx),
        all_concepts=covidpredict_dataset.get_concepts(),
        batch_size=BATCH_SIZE,
        include_ehr_token=best_params["include_ehr_token"],
        include_type_token=best_params["include_type_token"],
        include_vocab_token=best_params["include_vocab_token"],
        pooling_mode=(
            PoolingMode.MEAN
            if best_params["pooling_mode"] == "MEAN"
            else PoolingMode.CLS
        ),
        reduce_max_seq_length=False,
        add_dense_layer=best_params["add_dense_layer"],
        dense_layer_out_features=768,
        warmup_steps_multiplier=best_params["warmup_steps_multiplier"],
        num_epochs=20,  # best_params["num_epochs"],
        learning_rate=best_params["learning_rate"],
        num_eval_records=5000,
        verbose=True,
        include_evaluator=False,
    )

    best_model = output["model"]
    results = output["results"]

    print(f"Best params: {best_params}")

    def print_evaluation_results(results):
        accuracies = results["accuracies"]
        # metrics = results["metrics"]
        incorrect_predictions = results["incorrect_predictions"]

        print("Accuracies:")
        print(f"  Top-1 Accuracy: {accuracies['top_1_accuracy']:.3f}")
        print(f"  Top-5 Accuracy: {accuracies['top_5_accuracy']:.3f}")
        print(f"  Top-10 Accuracy: {accuracies['top_10_accuracy']:.3f}")
        print(f"  Top-100 Accuracy: {accuracies['top_100_accuracy']:.3f}")

    print_evaluation_results(results)

    # best_model.model.save(f"models/{EXPERIMENT_NAME}")


if __name__ == "__main__":
    run_experiment_two()
