import copy


def combine_datasets(*dataset_tuples):
    combined_concepts = []
    combined_parameters = []
    concept_offset = 0

    for params, concepts in dataset_tuples:
        # Append the concepts to the combined_concepts list
        combined_concepts.extend(concepts)

        # Adjust concept_idx in parameters and add to combined_parameters
        for param in params:
            new_param = copy.copy(param)
            if new_param.concept_idx is not None:
                new_param.concept_idx += concept_offset
            combined_parameters.append(new_param)

        # Update the concept_offset for the next iteration
        concept_offset += len(concepts)

    return combined_parameters, combined_concepts
