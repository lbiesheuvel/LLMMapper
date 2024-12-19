from src.data.sources.mapping_dataset import MappingDataset, Parameter, Concept
from src.data.enums import EHRToken, TypeToken, VocabularyToken
import pandas as pd
from sqlalchemy import create_engine
import logging


class ICUDataMappings(MappingDataset):
    def __init__(
        self,
        extract_after_last_semicolon,
        only_validated,
        only_active_concepts,
    ):
        self.extract_after_last_semicolon = extract_after_last_semicolon
        self.only_validated = only_validated
        self.engine = self._create_engine()
        self._parameters_df, self._concepts_df = self._load_data(only_active_concepts)
        super().__init__()

    def _create_engine(self):
        # Create the database engine
        POSTGRES_USER = # LEFT OUT FOR SECURITY REASONS
        POSTGRES_PASSWORD = # LEFT OUT FOR SECURITY REASONS
        POSTGRES_DB = # LEFT OUT FOR SECURITY REASONS
        DB_IP = # LEFT OUT FOR SECURITY REASONS
        DB_PORT = # LEFT OUT FOR SECURITY REASONS
        DB_URI = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_IP}:{DB_PORT}/{POSTGRES_DB}"
        engine = create_engine(DB_URI)
        return engine

    def _load_data(self, only_active_concepts: bool):
        # Load data from the database
        with self.engine.connect() as conn:
            parameters_df = pd.read_sql("SELECT * FROM parameter", conn)
            concepts_df = pd.read_sql("SELECT * FROM parameter_concept", conn)
            mappings_df = pd.read_sql("SELECT * FROM parameter_mapping", conn)

        # Preprocess dataframes
        parameters_df = self._preprocess_parameters(parameters_df)
        concepts_df = self._preprocess_concepts(concepts_df)
        mappings_df = self._preprocess_mappings(mappings_df)[
            ["parameter_id", "concept_id"]
        ]

        # Merge mappings into parameters
        parameters_df = pd.merge(
            parameters_df,
            mappings_df,
            left_on="id",
            right_on="parameter_id",
            how="left",
        )

        if only_active_concepts:
            mapped_concept_ids = parameters_df["concept_id"].dropna().unique()
            concepts_df = concepts_df[
                concepts_df["id"].isin(mapped_concept_ids)
            ].reset_index(drop=True)

        concepts_df["concept_index"] = concepts_df.index

        # Merge concepts into parameters
        parameters_df = pd.merge(
            parameters_df,
            concepts_df[["id", "concept_index"]],
            left_on="concept_id",
            right_on="id",
            how="left",
            suffixes=("_parameter", "_concept"),
        )

        return parameters_df, concepts_df

    def _preprocess_parameters(self, df):
        # Rename and process parameter dataframe
        df.rename(
            columns={
                "parameter_name": "name",
                "parameter_units": "units",
                "parameter_units_num": "units_num",
                "parameter_type": "type",
                "ehr_name": "source",
            },
            inplace=True,
        )
        df["source"] = df["source"].str.replace("metavision", "MV")
        df["source"] = df["source"].str.replace("epic", "EPIC")
        if self.extract_after_last_semicolon:
            df.loc[df["table"] == "orders", "name"] = df[df["table"] == "orders"][
                "name"
            ].apply(self._extract_after_last_semicolon)
        return df

    def _preprocess_concepts(self, df):
        df = df[df["valid_enddate"].isna()].reset_index(drop=True)
        return df

    def _preprocess_mappings(self, df):
        # Preprocess mappings dataframe
        df = df.sort_values(by=["parameter_id", "date_modified"], ascending=False)
        df = df.drop_duplicates(subset=["parameter_id", "user_id"], keep="first")
        if self.only_validated:
            # For each 'parameter_id', get the two most recent mappings
            df = df.groupby("parameter_id").head(2)
            # Keep mappings where 'parameter_id' and 'concept_id' are duplicated
            df = df[
                df.duplicated(
                    subset=["parameter_id", "parameter_concept_id"], keep="first"
                )
            ]
        else:
            # Keep the most recent mapping per 'parameter_id'
            df = df.groupby("parameter_id").head(1)
        df.rename(
            columns={"parameter_concept_id": "concept_id", "mapping_source": "source"},
            inplace=True,
        )
        df.reset_index(drop=True, inplace=True)
        return df

    def _get_ehr_token(self, ehr_name):
        if ehr_name == "EPIC":
            return EHRToken.EPIC
        elif ehr_name == "HIX":
            return EHRToken.HIX
        elif ehr_name == "MV":
            return EHRToken.MV
        else:
            raise ValueError(f"Unknown EHR name: {ehr_name}")

    def _get_vocab_token(self, vocab_name):
        if vocab_name == "LOINC":
            return VocabularyToken.LOINC
        elif vocab_name == "RXNORM":
            return VocabularyToken.RXNORM
        elif vocab_name == "ICUDATA":
            return VocabularyToken.ICUDATA
        elif vocab_name == None:
            return VocabularyToken.ICUDATA
        else:
            raise ValueError(f"Unknown vocabulary name: {vocab_name}")

    def is_medication_parameter(self, row):
        # LET OP: Werkt niet voor UMCU! Die zullen false zijn
        return row["table"] == "medications" or row["type"] in [
            "medications",
            "Medicatie",
        ]

    def is_medication_concept(self, row):
        return row["category_main"] == "medication"

    def filter(
        self,
        only_mapped=False,
        exclude_irrelevant=False,
        exclude_relevant_but_no_existing_concept=False,
        hospital_name=None,
    ):
        # Apply filters to the parameters dataframe
        df = self._parameters_df.copy()

        if only_mapped:
            df = df[df["concept_id"].notna()]

        if exclude_relevant_but_no_existing_concept:
            df = df[
                ~df["concept_id"].isin(
                    [
                        726,
                        5093,
                        41698,
                    ]
                )
            ]

        if exclude_irrelevant:
            # irrelevant means that concept id is 725
            df = df[df["concept_id"] != 725]

        if hospital_name is not None:
            df = df[df["hospital_name"] == hospital_name]

        df.reset_index(drop=True, inplace=True)
        self._parameters_df = df

    def get_parameters(self, indices=None):
        parameters_df = self._parameters_df.copy()

        if indices is not None:
            parameters_df = parameters_df.loc[indices]

        parameters = [
            Parameter(
                parameter_id=(
                    int(row["id_parameter"]) if pd.notna(row["id_parameter"]) else None
                ),
                name=row["name"],
                table=row["table"],
                type_token=(
                    TypeToken.MEDICATION
                    if self.is_medication_parameter(row)
                    else TypeToken.OTHERS
                ),
                hospital_name=row["hospital_name"],
                ehr_token=self._get_ehr_token(row["source"]),
                unit=row["units"],
                concept_idx=(
                    int(row["concept_index"])
                    if pd.notna(row["concept_index"])
                    else None
                ),  # Use concept_index
            )
            for _, row in parameters_df.iterrows()
        ]
        return parameters

    def get_concepts(self):
        concepts = [
            Concept(
                concept_id=(int(row["id"]) if pd.notna(row["id"]) else None),
                name=row["name"],
                category=row["category_main"],
                vocabulary_token=self._get_vocab_token(row["vocabulary_id"]),
                type_token=(
                    TypeToken.MEDICATION
                    if self.is_medication_concept(row)
                    else TypeToken.OTHERS
                ),
            )
            for _, row in self._concepts_df.iterrows()
        ]
        return concepts

    def _extract_after_last_semicolon(self, input_string):
        # Extract substring after the last semicolon
        parts = input_string.split(";")
        return parts[-1].strip() if parts else ""
