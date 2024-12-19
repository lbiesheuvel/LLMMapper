import random

EXPERIMENT_NAME = "covidpredict"
MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 32
SEED = 42


from src.data.sources.covidpredict_mappings import CovidPredictMappings
from src.data.sources.icudata_mappings import ICUDataMappings
from src.data.enums import VocabularyToken, PoolingMode
from src.sentence_transformers_model.train.MappingModel import MappingModel
from src.sentence_transformers_model.train.CustomTrainEvaluator import (
    CustomTrainEvaluator,
)


def run_experiment_three():
    covidpredict_dataset = CovidPredictMappings(
        extract_after_last_semicolon=True, only_active_concepts=True, limited_mode=True
    )
    covidpredict_dataset.filter(only_mapped=True)

    covidpredict_parameters = covidpredict_dataset.get_parameters()
    covidpredict_concepts = covidpredict_dataset.get_concepts()

    mapping_db_dataset = ICUDataMappings(
        extract_after_last_semicolon=True,
        only_validated=True,
        only_active_concepts=False,
    )

    mapping_db_dataset.filter(
        only_mapped=True,
        exclude_irrelevant=True,
        exclude_relevant_but_no_existing_concept=True,
    )

    mapping_db_parameters = mapping_db_dataset.get_parameters()
    mapping_db_concepts = mapping_db_dataset.get_concepts()

    forbidden_combinations = [
        [VocabularyToken.ICUNITY, VocabularyToken.ICUDATA, VocabularyToken.LOINC],
        [VocabularyToken.RXNORM, VocabularyToken.ATC],
    ]

    model = MappingModel(
        model_name=MODEL_NAME,
        include_query_text=True,
        include_ehr_token=True,
        include_type_token=True,
        include_vocab_token=False,
        pooling_mode=PoolingMode.MEAN,
        reduce_max_seq_length=False,
        add_dense_layer=False,
        dense_layer_out_features=None,
    )

    evaluator = CustomTrainEvaluator(
        trial=None,
        test_parameters=mapping_db_parameters,
        all_concepts=mapping_db_concepts,
        include_query_text=True,
        include_ehr_token=True,
        include_type_token=True,
        include_vocab_token=False,
        show_progress_bar=True,
        print_results=True,
        num_test_records=5000,
    )

    model.train(
        lr=3.4062012929735624e-05,
        batch_size=BATCH_SIZE,
        num_epochs=20,
        warmup_steps_multiplier=0.0959993538249428,
        evaluator=evaluator,
        train_parameters=covidpredict_parameters,
        all_concepts=covidpredict_concepts,
        forbidden_combinations=forbidden_combinations,
        show_progress_bar=True,
        verbose=True,
    )

    output = model.evaluate(
        test_parameters=mapping_db_parameters,
        all_concepts=mapping_db_concepts,
        show_progress_bar=True,
    )

    def print_evaluation_results(results):
        accuracies = results["accuracies"]
        incorrect_predictions = results["incorrect_predictions"]

        print("Accuracies:")
        print(f"  Top-1 Accuracy: {accuracies['top_1_accuracy']:.2f}")
        print(f"  Top-5 Accuracy: {accuracies['top_5_accuracy']:.2f}")
        print(f"  Top-10 Accuracy: {accuracies['top_10_accuracy']:.2f}")
        print(f"  Top-100 Accuracy: {accuracies['top_100_accuracy']:.2f}")

        # Randomly select up to 100 incorrect predictions
        num_mistakes = min(100, len(incorrect_predictions))
        random_mistakes = random.sample(incorrect_predictions, num_mistakes)

        print(f"\n{num_mistakes} Random Incorrect Predictions:")
        for i, (query, true_concept, predicted_concepts) in enumerate(
            random_mistakes, 1
        ):
            print(f"{i}.")
            print(f"  Query: {query}")
            print(f"  True Concept: {true_concept}")
            print(f"  Predicted Concepts: {', '.join(predicted_concepts)}\n")

    print_evaluation_results(output)


if __name__ == "__main__":
    run_experiment_three()
