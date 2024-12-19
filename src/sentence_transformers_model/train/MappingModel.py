import numpy as np
from sentence_transformers import SentenceTransformer, losses
from src.data.enums import PoolingMode, VocabularyToken
from sentence_transformers.models import Normalize, Dense
from torch.nn import Tanh
from src.sentence_transformers_model.train.CustomDataloader import (
    CustomNoDuplicatesDataloader,
)
from typing import List, Tuple, Optional, Dict, Any
from sentence_transformers.evaluation import SentenceEvaluator
import re
from src.data.sources.mapping_dataset import Parameter, Concept
from src.sentence_transformers_model.helpers import get_input_examples
import torch


class MappingModel:

    def __init__(
        self,
        model_name: str,
        include_query_text: bool,
        include_ehr_token: bool,
        include_type_token: bool,
        include_vocab_token: bool,
        pooling_mode: PoolingMode,
        reduce_max_seq_length: bool,
        add_dense_layer: bool,
        dense_layer_out_features: int,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.include_query_text = include_query_text
        self.include_ehr_token = include_ehr_token
        self.include_type_token = include_type_token
        self.include_vocab_token = include_vocab_token
        word_embedding_model = self.model._first_module()
        if reduce_max_seq_length:
            word_embedding_model.max_seq_length = 64

        if pooling_mode == "CLS":
            self.model._modules["1"].pooling_mode_mean_tokens = False
            self.model._modules["1"].pooling_mode_cls_token = True

        # Check if there is a normalization layer (last module)), if not, add it as a last module
        # first check if module "2"
        if "2" in self.model._modules:
            if isinstance(self.model._modules["2"], Normalize):
                # remove the normalization layer
                self.model._modules.pop("2")

        if add_dense_layer:
            dense_model = Dense(
                in_features=self.model.get_sentence_embedding_dimension(),
                out_features=dense_layer_out_features,
                activation_function=Tanh(),
            )
            self.model.add_module("2", dense_model)

        self.model.add_module("3" if add_dense_layer else "2", Normalize())

        self.trained_tokens = set()

    def train(
        self,
        lr: float,
        batch_size: int,
        num_epochs: int,
        warmup_steps_multiplier: float,
        evaluator: Optional[SentenceEvaluator],
        train_parameters: List[Parameter],
        all_concepts: List[Concept],
        forbidden_combinations: Optional[List[List[VocabularyToken]]],
        show_progress_bar: bool,
        verbose: bool,
    ):

        train_parameter_queries = [
            parameter.generate_llm_query(
                include_query_text=self.include_query_text,
                include_par_token=True,
                include_ehr_token=self.include_ehr_token,
                include_units=True,
                include_type_token=self.include_type_token,
            )
            for parameter in train_parameters
        ]
        train_concept_queries = [
            all_concepts[parameter.concept_idx].generate_llm_query(
                include_query_text=self.include_query_text,
                include_con_token=True,
                include_vocabulary_token=self.include_vocab_token,
                include_type_token=self.include_type_token,
            )
            for parameter in train_parameters
        ]

        vocab_tokens = [
            all_concepts[parameter.concept_idx].vocabulary_token
            for parameter in train_parameters
        ]

        train_samples, vocabulary = get_input_examples(
            parameter_queries=train_parameter_queries,
            concept_queries=train_concept_queries,
            vocabulary_tokens=vocab_tokens,
        )

        # Iterate over the train_samples to extract the tokens and add them to the trained model, so that we know which tokens are trained
        for example in train_samples:
            for text in example.texts:
                # we extract all tokens between [ and ] and add them to the trained tokens
                tokens = self.__extract_tokens(text)
                self.trained_tokens.update(tokens)

        if len(self.trained_tokens) > 0:
            word_embedding_model = self.model._first_module()
            word_embedding_model.tokenizer.add_tokens(list(self.trained_tokens))
            word_embedding_model.auto_model.resize_token_embeddings(
                len(word_embedding_model.tokenizer)
            )
            if verbose:
                print(
                    f"Added {len(self.trained_tokens)} tokens to the model: {', '.join(self.trained_tokens)}"
                )

        train_loss = losses.MultipleNegativesRankingLoss(model=self.model)

        train_dataloader = CustomNoDuplicatesDataloader(
            train_examples=train_samples,
            vocabulary=vocabulary,
            batch_size=batch_size,
            forbidden_combinations=forbidden_combinations,
        )

        warmup_steps = int(len(train_dataloader) * num_epochs * warmup_steps_multiplier)
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": lr},
            use_amp=True,
            show_progress_bar=show_progress_bar,
            evaluator=evaluator,
        )

    def evaluate(
        self,
        test_parameters: List[Parameter],
        all_concepts: List[Concept],
        show_progress_bar: bool = False,
        batch_size: int = 64,
    ) -> Dict[str, Any]:

        # Put model in evaluation mode
        self.model.eval()

        test_parameter_queries = [
            parameter.generate_llm_query(
                include_query_text=self.include_query_text,
                include_par_token=True,
                include_ehr_token=self.include_ehr_token,
                include_units=True,
                include_type_token=self.include_type_token,
            )
            for parameter in test_parameters
        ]

        all_concept_queries = [
            concept.generate_llm_query(
                include_query_text=self.include_query_text,
                include_con_token=True,
                include_vocabulary_token=self.include_vocab_token,
                include_type_token=self.include_type_token,
            )
            for concept in all_concepts
        ]

        # Iterate over the parameter_queries to extract the tokens to check if they are trained, else, throw an error
        for example in test_parameter_queries + all_concept_queries:
            # we extract all tokens between [ and ] and check if they are in the trained tokens
            tokens = self.__extract_tokens(example)
            for token in tokens:
                if token not in self.trained_tokens:
                    raise ValueError(
                        f"Token {token} in parameter query {example} is not trained"
                    )

        parameter_embeddings = self.model.encode(
            test_parameter_queries,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
        )
        concept_embeddings = self.model.encode(
            all_concept_queries,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
        )
        concept_embeddings = np.array(concept_embeddings)
        concept_embeddings_norm = concept_embeddings / np.linalg.norm(
            concept_embeddings, axis=1, keepdims=True
        )
        all_concept_names = [concept.name for concept in all_concepts]

        # Process parameter embeddings in batches
        top_k = 100
        top_similar_concept_names = []
        incorrect_predictions = []
        top_1_accuracy = 0
        top_5_accuracy = 0
        top_10_accuracy = 0
        top_100_accuracy = 0

        test_concept_names = [
            all_concepts[parameter.concept_idx].name for parameter in test_parameters
        ]

        from torch.utils.data import DataLoader, TensorDataset

        parameter_dataset = TensorDataset(torch.tensor(parameter_embeddings))
        parameter_loader = DataLoader(parameter_dataset, batch_size=batch_size)

        for batch_idx, (batch_embeddings,) in enumerate(parameter_loader):
            batch_embeddings = batch_embeddings.numpy()
            batch_embeddings_norm = batch_embeddings / np.linalg.norm(
                batch_embeddings, axis=1, keepdims=True
            )
            cosine_similarity = np.dot(batch_embeddings_norm, concept_embeddings_norm.T)
            top_indices = np.argsort(-cosine_similarity, axis=1)[:, :top_k]
            batch_top_similar_concept_names = [
                [all_concept_names[j] for j in indices] for indices in top_indices
            ]

            # Calculate accuracies for the current batch
            for i, idx in enumerate(
                range(
                    batch_idx * batch_size,
                    min((batch_idx + 1) * batch_size, len(test_concept_names)),
                )
            ):
                true_concept = test_concept_names[idx]
                predicted_concepts = batch_top_similar_concept_names[i]

                if true_concept in predicted_concepts[:1]:
                    top_1_accuracy += 1
                if true_concept in predicted_concepts[:5]:
                    top_5_accuracy += 1
                else:
                    incorrect_predictions.append(
                        (
                            test_parameter_queries[idx],
                            true_concept,
                            predicted_concepts[:5],
                        )
                    )
                if true_concept in predicted_concepts[:10]:
                    top_10_accuracy += 1
                if true_concept in predicted_concepts[:top_k]:
                    top_100_accuracy += 1

            top_similar_concept_names.extend(batch_top_similar_concept_names)

        # Calculate final accuracies
        total_samples = len(test_concept_names)
        top_1_accuracy /= total_samples
        top_5_accuracy /= total_samples
        top_10_accuracy /= total_samples
        top_100_accuracy /= total_samples

        return {
            "accuracies": {
                "top_1_accuracy": top_1_accuracy,
                "top_5_accuracy": top_5_accuracy,
                "top_10_accuracy": top_10_accuracy,
                "top_100_accuracy": top_100_accuracy,
            },
            "incorrect_predictions": incorrect_predictions,
        }

    def __extract_tokens(self, s: str) -> list:
        pattern = r"\[-[A-Za-z0-9_-]+-\]"
        tokens = re.findall(pattern, s)
        return tokens
