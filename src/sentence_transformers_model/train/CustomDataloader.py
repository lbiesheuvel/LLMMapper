from sentence_transformers import datasets
import random

# Imports for type inference:
from sentence_transformers import InputExample
from typing import List, Optional
from src.data.enums import VocabularyToken


class CustomNoDuplicatesDataloader(datasets.NoDuplicatesDataLoader):

    def __init__(
        self,
        train_examples: List[InputExample],
        vocabulary: List[VocabularyToken],
        batch_size: int,
        forbidden_combinations: Optional[List[List[VocabularyToken]]],
    ):
        self.batch_size = batch_size
        self.data_pointer = 0
        self.collate_fn = None
        self.train_examples = list(zip(train_examples, vocabulary))
        random.shuffle(self.train_examples)
        # forbidden combinations is a list of list of forbidden combinations of vocabulary names
        self.forbidden_combinations = forbidden_combinations

    def __iter__(self):
        for _ in range(self.__len__()):
            batch = []
            texts_in_batch = set()
            vocab_in_batch = set()
            while len(batch) < self.batch_size:
                example = self.train_examples[self.data_pointer]

                valid_example = True
                for text in example[0].texts:
                    if text.strip().lower() in texts_in_batch:
                        valid_example = False
                        break

                if len(vocab_in_batch) > 0:
                    if example[1] not in vocab_in_batch:
                        # Check if the current example has a concept that is in the same vocabulary as the previous example in the batch. If not, skip this example
                        for forbidden_vocab_set in self.forbidden_combinations:
                            any_of_combinations_in_batch = any(
                                vocab in forbidden_vocab_set for vocab in vocab_in_batch
                            )
                            current_vocab_in_combinations = (
                                example[1] in forbidden_vocab_set
                            )
                            if (
                                any_of_combinations_in_batch
                                and current_vocab_in_combinations
                            ):
                                valid_example = False
                                break

                if valid_example:
                    vocab_in_batch.add(example[1])
                    batch.append(example[0])
                    for text in example[0].texts:
                        texts_in_batch.add(text.strip().lower())

                self.data_pointer += 1
                if self.data_pointer >= len(self.train_examples):
                    self.data_pointer = 0
                    random.shuffle(self.train_examples)

            yield self.collate_fn(batch) if self.collate_fn is not None else batch
