from sentence_transformers.evaluation import SentenceEvaluator

from optuna import TrialPruned

from typing import List, Optional
from optuna import Trial
import numpy as np
from src.data.sources.mapping_dataset import Parameter, Concept


class CustomTrainEvaluator(SentenceEvaluator):

    def __init__(
        self,
        trial: Optional[Trial],
        test_parameters: List[Parameter],
        all_concepts: List[Concept],
        include_query_text: bool,
        include_ehr_token: bool,
        include_type_token: bool,
        include_vocab_token: bool,
        show_progress_bar: bool,
        print_results: bool,
        num_test_records: Optional[int],
    ):
        self.trial = trial
        self.all_concepts = all_concepts
        self.show_progress_bar = show_progress_bar
        self.print_results = print_results
        self.include_ehr_token = include_ehr_token
        self.include_type_token = include_type_token
        self.include_vocab_token = include_vocab_token
        self.include_query_text = include_query_text
        if num_test_records is not None:
            # We randomize the test records (a list) and select the first num_test_records. Both test_parameters and test_concepts are randomized in the same way because they are related
            randomization = np.random.RandomState(42)
            randomization.shuffle(test_parameters)
            num_test_records = min(len(test_parameters), num_test_records)
            self.test_parameters = test_parameters[:num_test_records]
            self.test_concepts = [
                all_concepts[parameter.concept_idx]
                for parameter in self.test_parameters
            ]

        else:
            self.test_parameters = test_parameters
            self.test_concepts = [
                all_concepts[parameter.concept_idx]
                for parameter in self.test_parameters
            ]

    def __call__(
        self, model, output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> float:

        test_parameter_queries = [
            parameter.generate_llm_query(
                include_query_text=self.include_query_text,
                include_par_token=True,
                include_ehr_token=self.include_ehr_token,
                include_units=True,
                include_type_token=self.include_type_token,
            )
            for parameter in self.test_parameters
        ]

        all_concept_queries = [
            concept.generate_llm_query(
                include_query_text=self.include_query_text,
                include_con_token=True,
                include_vocabulary_token=self.include_vocab_token,
                include_type_token=self.include_type_token,
            )
            for concept in self.all_concepts
        ]

        parameter_embeddings = model.encode(
            test_parameter_queries, show_progress_bar=self.show_progress_bar
        ).tolist()
        concept_embeddings = model.encode(
            all_concept_queries, show_progress_bar=self.show_progress_bar
        ).tolist()
        parameter_embeddings = np.array(parameter_embeddings)
        concept_embeddings = np.array(concept_embeddings)
        array1_norm = parameter_embeddings / np.linalg.norm(
            parameter_embeddings, axis=1, keepdims=True
        )
        array2_norm = concept_embeddings / np.linalg.norm(
            concept_embeddings, axis=1, keepdims=True
        )
        cosine_similarity = np.dot(array1_norm, array2_norm.T)
        top_similar_concepts = np.argsort(-cosine_similarity, axis=1)[:, :10]
        all_concept_names = [concept.name for concept in self.all_concepts]
        top_similar_concept_names = [
            [all_concept_names[j] for j in i] for i in top_similar_concepts
        ]

        # we calculate the top 1, 5 and 10 accuracy
        incorrect_predictions = []
        top_1_accuracy = 0
        top_5_accuracy = 0
        top_10_accuracy = 0

        test_concept_names = [concept.name for concept in self.test_concepts]
        for i in range(len(test_concept_names)):
            if test_concept_names[i] in top_similar_concept_names[i][:1]:
                top_1_accuracy += 1
            if test_concept_names[i] in top_similar_concept_names[i][:5]:
                top_5_accuracy += 1
            else:
                incorrect_predictions.append(
                    (
                        test_parameter_queries[i],
                        test_concept_names[i],
                        top_similar_concept_names[i][:5],
                    )
                )
            if test_concept_names[i] in top_similar_concept_names[i][:10]:
                top_10_accuracy += 1
        top_1_accuracy /= len(test_concept_names)
        top_5_accuracy /= len(test_concept_names)
        top_10_accuracy /= len(test_concept_names)

        if self.print_results:
            print(
                f"Top 1 accuracy: {top_1_accuracy}, Top 5 accuracy: {top_5_accuracy}, Top 10 accuracy: {top_10_accuracy}"
            )
        if self.trial is None:
            return top_5_accuracy
        # report to optuna
        self.trial.report(top_5_accuracy, step=epoch)
        # Handle pruning based on the intermediate value.
        if self.trial.should_prune():
            raise TrialPruned()
        return top_5_accuracy
