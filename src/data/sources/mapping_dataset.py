# mapping_dataset.py
import pandas as pd
from abc import ABC, abstractmethod
from src.data.enums import EHRToken, TypeToken, VocabularyToken
from typing import List, Optional
import logging


class Parameter:
    def __init__(
        self,
        name: str,
        table: str,
        type_token: TypeToken,
        hospital_name: str,
        ehr_token: EHRToken,
        unit: str,
        parameter_id: Optional[int] = None,
        concept_idx: Optional[int] = None,
    ):
        self.parameter_id = parameter_id
        self.name = name
        self.table = table
        self.type_token = type_token
        self.hospital_name = hospital_name
        self.ehr_token = ehr_token
        self.unit = unit
        self.concept_idx = concept_idx

    def __str__(self):
        return f"Parameter: {self.name}, Parameter Table: {self.table}, Type Token: {self.type_token.value if self.type_token else ''}, Hospital Name: {self.hospital_name}, EHR Token: {self.ehr_token.value if self.ehr_token else ''}, Unit: {self.unit}, Concept Index: {self.concept_idx}"

    def generate_llm_query(
        self,
        include_query_text: bool,
        include_par_token: bool,
        include_ehr_token: bool,
        include_units: bool,
        include_type_token: bool,
    ):
        query = f"query: " if include_query_text else f""
        query += f"[-PAR-]" if include_par_token else f""
        query += (
            f"{self.type_token.value}"
            if (include_type_token and self.type_token)
            else ""
        )
        query += (
            f"{self.ehr_token.value}" if (include_ehr_token and self.ehr_token) else ""
        )

        query += f"{self.name}".lower()
        query += f"[-UNIT-]{self.unit}" if (include_units and self.unit) else ""

        return query


class Concept:
    def __init__(
        self,
        name: str,
        category: str,
        vocabulary_token: VocabularyToken,
        type_token: TypeToken,
        concept_id: Optional[int] = None,
    ):
        self.concept_id = concept_id
        self.name = name
        self.category = category
        self.vocabulary_token = vocabulary_token
        self.type_token = type_token

    def __str__(self):
        return f"Concept: {self.name}, Concept Category: {self.category}, Vocabulary Token: {self.vocabulary_token.value if self.vocabulary_token else ''}, Type Token: {self.type_token.value if self.type_token else ''}"

    def generate_llm_query(
        self,
        include_query_text: bool,
        include_con_token: bool,
        include_vocabulary_token: bool,
        include_type_token: bool,
    ) -> str:
        query = f"query: " if include_query_text else f""
        query += f"[-CON-]" if include_con_token else f""
        query += (
            f"{self.type_token.value}"
            if (include_type_token and self.type_token)
            else ""
        )
        query += (
            f"{self.vocabulary_token.value}"
            if (include_vocabulary_token and self.vocabulary_token)
            else ""
        )
        query += f"{self.name}".lower()

        return query


class MappingDataset(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_parameters(self, indices=None):
        pass

    @abstractmethod
    def get_concepts(self):
        pass
