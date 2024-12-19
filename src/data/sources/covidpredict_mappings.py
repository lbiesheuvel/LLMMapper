# covidpredict_mappings.py
from src.data.sources.mapping_dataset import MappingDataset, Parameter, Concept
from src.data.enums import EHRToken, TypeToken, VocabularyToken
import pandas as pd
import ast
import logging
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


class CovidPredictMappings(MappingDataset):
    def __init__(
        self,
        extract_after_last_semicolon: bool,
        only_active_concepts: bool,
        limited_mode,
    ):

        self._mappings_df, self._concepts_df = self._load_data(
            extract_after_last_semicolon, only_active_concepts
        )
        self.limited_mode = limited_mode

        super().__init__()

    def _load_data(
        self, extract_after_last_semicolon: bool, only_active_concepts: bool
    ):
        try:
            mappings_file_path = "src\data\sources\covidpredict\mappings.csv"
            concepts_file_path = "src\data\sources\covidpredict\concepts.csv"
            mappings_df = pd.read_csv(mappings_file_path)

            mappings_df["unit"] = mappings_df["unit"].apply(self._process_units)

            if extract_after_last_semicolon:
                mappings_df.loc[mappings_df["table"] == "orders", "parameter_name"] = (
                    mappings_df[mappings_df["table"] == "orders"][
                        "parameter_name"
                    ].apply(self._extract_after_last_semicolon)
                )

            mappings_df = mappings_df.drop_duplicates(
                subset=["parameter_name", "ehr_name", "hospital_name", "table"]
            ).reset_index(drop=True)

            logging.info(f"Data geladen uit {mappings_file_path}")
            concepts_df = pd.read_csv(concepts_file_path)

            # drop row where concept_name is unmapped
            concepts_df = concepts_df[
                concepts_df["concept_label"] != "unmapped"
            ].reset_index(drop=True)

            if only_active_concepts:
                concepts_df = concepts_df[
                    concepts_df["concept_label"].isin(mappings_df["concept_label"])
                ].reset_index(drop=True)

            concepts_df["concept_index"] = concepts_df.index

            mappings_df = mappings_df.merge(
                concepts_df,
                how="left",
                left_on="concept_label",
                right_on="concept_label",
            )

            logging.info(f"Concepten geladen uit {concepts_file_path}")
            return mappings_df, concepts_df
        except Exception as e:
            logging.error(f"Fout bij het laden van data: {e}")
            raise

    def filter(self, hospital_name: str = None, only_mapped: bool = False):
        if hospital_name:
            self._mappings_df = self._mappings_df[
                self._mappings_df["hospital_name"] == hospital_name
            ]
            logging.info(f"Gefilterd op ziekenhuisnaam: {hospital_name}")
        if only_mapped:
            self._mappings_df = self._mappings_df[
                self._mappings_df["concept_label"] != "unmapped"
            ]
            logging.info("Gefilterd op alleen gemapte parameters")
        self._mappings_df.reset_index(drop=True, inplace=True)

    def _get_type_token_parameter(self, table_name, limited_mode):
        if limited_mode:
            if table_name == "medications":
                return TypeToken.MEDICATION
            else:
                return TypeToken.OTHERS
        else:
            if table_name == "medications":
                return TypeToken.MEDICATION
            elif table_name == "measurements":
                return TypeToken.MEASUREMENTS
            elif table_name == "orders":
                return TypeToken.ORDERS
            elif table_name == "range_signals_joined":
                return TypeToken.RANGE_SIGNALS_JOINED
            elif table_name == "events":
                return TypeToken.EVENTS
            elif table_name == "fluid_in":
                return TypeToken.FLUID_IN
            elif table_name == "fluid_out":
                return TypeToken.FLUID_OUT
            elif table_name == "lda":
                return TypeToken.LDA
            else:
                return None

    def _get_type_token_concept(self, category_name, limited_mode):
        if limited_mode:
            if category_name == "medication":
                return TypeToken.MEDICATION
            else:
                return TypeToken.OTHERS
        else:
            if category_name == "medication":
                return TypeToken.MEDICATION
            elif category_name == "laboratory value":
                return TypeToken.LABORATORY
            elif category_name == "hemodynamics":
                return TypeToken.HEMODYNAMICS
            elif category_name == "fluid balance":
                return TypeToken.FLUID_BALANCE
            elif category_name == "lda":
                return TypeToken.LDA
            elif category_name == "infectiology":
                return TypeToken.INFECTIOLOGY
            elif category_name == "respiratory":
                return TypeToken.RESPIRATORY
            elif category_name == "neurology":
                return TypeToken.NEUROLOGY
            elif category_name == "renal replacement therapy":
                return TypeToken.RENAL_REPLACEMENT_THERAPY
            elif category_name == "admission information":
                return TypeToken.ADMISSION_INFORMATION
            elif category_name == "clinical score":
                return TypeToken.CLINICAL_SCORE
            elif category_name == "nice data":
                return TypeToken.NICE_DATA
            elif category_name == "demographics":
                return TypeToken.DEMOGRAPHICS
            elif category_name == "position":
                return TypeToken.POSITION
            elif category_name == "sofa score":
                return TypeToken.SOFA_SCORE
            else:
                return None

    def _get_ehr_token(self, ehr_name):
        if ehr_name == "EPIC":
            return EHRToken.EPIC
        elif ehr_name == "HIX":
            return EHRToken.HIX
        elif ehr_name == "MV":
            return EHRToken.MV

    def get_parameters(self, indices=None):
        mappings_df = self._mappings_df.copy()
        if indices is not None:
            mappings_df = mappings_df.loc[indices]

        parameters = [
            Parameter(
                name=row["parameter_name"],
                table=row["table"],
                type_token=self._get_type_token_parameter(
                    row["table"], self.limited_mode
                ),
                hospital_name=row["hospital_name"],
                ehr_token=self._get_ehr_token(row["ehr_name"]),
                unit=row["unit"],
                concept_idx=(
                    int(row["concept_index"])
                    if not pd.isna(row["concept_index"])
                    else None
                ),
            )
            for _, row in mappings_df.iterrows()
        ]

        return parameters

    def get_concepts(self):
        concepts = [
            Concept(
                name=row["concept_label"],
                category=row["category"],
                vocabulary_token=VocabularyToken.ICUNITY,
                type_token=self._get_type_token_concept(
                    row["category"], self.limited_mode
                ),
            )
            for _, row in self._concepts_df.iterrows()
        ]
        return concepts

    def get_tariq_splits(self):
        # training data: amc, vumc, erasmus and olvg
        # testing data: all other hospitals
        all_parameters = self._mappings_df
        train_hospitals = ["aumc", "vumc", "erasmus", "olvg"]
        # get indices of of train set (where hospital_name is in train_hospitals) and test set (where hospital_name is not in train_hospitals)
        train_indices = all_parameters[
            all_parameters["hospital_name"].isin(train_hospitals)
        ].index
        test_indices = all_parameters[
            ~all_parameters["hospital_name"].isin(train_hospitals)
        ].index
        return train_indices, test_indices

    def get_splits(
        self,
        n_hospitals_first_split,
        n_hospitals_second_split,
        first_split_ehr,
        second_split_ehr,
        seed,
        starting_idx=None,
    ):
        np.random.seed(seed)
        all_parameters = self._mappings_df.copy()
        if starting_idx is not None:
            all_parameters = all_parameters.loc[starting_idx]

        unique_hospitals = all_parameters[
            ["hospital_name", "ehr_name"]
        ].drop_duplicates()

        if first_split_ehr != "MIX":
            train_hospitals = unique_hospitals[
                unique_hospitals["ehr_name"] == first_split_ehr
            ]
        else:
            ehrs = unique_hospitals["ehr_name"].unique()
            per_ehr = max(1, n_hospitals_first_split // len(ehrs))
            train_hospitals = pd.DataFrame()
            for ehr in ehrs:
                ehr_hospitals = unique_hospitals[
                    unique_hospitals["ehr_name"] == ehr
                ].sample(
                    n=min(
                        per_ehr,
                        len(unique_hospitals[unique_hospitals["ehr_name"] == ehr]),
                    ),
                    random_state=seed,
                )
                train_hospitals = pd.concat(
                    [train_hospitals, ehr_hospitals], ignore_index=True
                )

        if second_split_ehr != "MIX":
            test_hospitals = unique_hospitals[
                (unique_hospitals["ehr_name"] == second_split_ehr)
                & (
                    ~unique_hospitals["hospital_name"].isin(
                        train_hospitals["hospital_name"]
                    )
                )
            ]
        else:
            ehrs = unique_hospitals["ehr_name"].unique()
            per_ehr = max(1, n_hospitals_second_split // len(ehrs))
            test_hospitals = pd.DataFrame()
            for ehr in ehrs:
                ehr_hospitals = unique_hospitals[
                    (unique_hospitals["ehr_name"] == ehr)
                    & (
                        ~unique_hospitals["hospital_name"].isin(
                            train_hospitals["hospital_name"]
                        )
                    )
                ]
                ehr_hospitals_sample = ehr_hospitals.sample(
                    n=min(per_ehr, len(ehr_hospitals)), random_state=seed
                )
                test_hospitals = pd.concat(
                    [test_hospitals, ehr_hospitals_sample], ignore_index=True
                )

        train_hospitals_selected = train_hospitals.sample(
            n=min(n_hospitals_first_split, len(train_hospitals)), random_state=seed
        )
        test_hospitals_selected = test_hospitals.sample(
            n=min(n_hospitals_second_split, len(test_hospitals)), random_state=seed
        )

        indices_first_split = all_parameters[
            all_parameters["hospital_name"].isin(
                train_hospitals_selected["hospital_name"]
            )
        ].index
        indices_second_split = all_parameters[
            all_parameters["hospital_name"].isin(
                test_hospitals_selected["hospital_name"]
            )
        ].index

        return indices_first_split, indices_second_split

    def get_proportional_splits(self, p_test, seed):

        all_parameters = self._mappings_df.copy()
        unique_hospitals = all_parameters[
            ["hospital_name", "ehr_name"]
        ].drop_duplicates()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=p_test, random_state=seed)
        for train_index, test_index in sss.split(
            unique_hospitals, unique_hospitals["ehr_name"]
        ):
            train_hospitals = unique_hospitals.iloc[train_index]
            test_hospitals = unique_hospitals.iloc[test_index]

        train_indices = all_parameters[
            all_parameters["hospital_name"].isin(train_hospitals["hospital_name"])
        ].index
        test_indices = all_parameters[
            all_parameters["hospital_name"].isin(test_hospitals["hospital_name"])
        ].index

        return train_indices, test_indices

    def _process_units(self, unit: str) -> str:
        unit = unit.replace("nan", "'Remove'")
        unit = unit.replace("None", "'Remove'")
        unit = unit.replace("Geen", "")
        unit = ast.literal_eval(unit)
        unit = [x for x in unit if x not in ["Remove", ""]]
        return ", ".join(unit)

    def _extract_after_last_semicolon(self, input_string: str) -> str:
        parts = input_string.split(";")
        return parts[-1].strip() if parts else ""
