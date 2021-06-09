import numpy as np
import pandas as pd
import pickle


def correlation(vector, matrix):
    vector1 = vector - np.mean(vector)
    matrix1 = matrix - np.mean(matrix, 1).reshape((-1, 1))
    corr1 = np.sum(vector1 * matrix1, 1) / np.sqrt(np.sum(vector1 * vector1))
    return corr1 / np.sqrt(np.sum(matrix1 * matrix1, axis=1))


def finding_corr_threshold(corr_threshold):
    # loading database
    print('loading database...')
    with open('COALS_data.bin', 'rb') as f:
        [word_list, table7_matrix] = pickle.loads(f.read())
    with open('Identifying_result.bin', 'rb') as f:
        [nouns, adjectives_vervs] = pickle.loads(f.read())
    print('\tfinished loading database.')

    # COALS vectors for nouns
    COALS_vectors = {}
    for item in nouns:
        COALS_vectors[item] = table7_matrix[word_list.index(item), :]
    # COALS vectors for adjectives + verbs
    COALS_matrix = []
    adjectives_verbs = []
    for item in adjectives_vervs:
        COALS_matrix.append(table7_matrix[word_list.index(item), :])
        adjectives_verbs.append(item)

    COALS_matrix = np.asarray(COALS_matrix, np.float32)
    adjectives_verbs = np.asarray(adjectives_verbs, np.str)
    print('There are made {} noun vectors and {} (adjective + verb) vectors'.format(len(nouns), len(adjectives_vervs)))

    print('Finding the adjectives and verbs similar to each noun by correlation threshold {}'.format(corr_threshold))
    result_noun = []
    result_similars = []
    for noun in nouns:
        co = correlation(COALS_vectors[noun], COALS_matrix)
        indices = np.where(co >= corr_threshold)
        result_str = ''
        for str_item in adjectives_verbs[indices]:
            if str_item != noun:
                result_str += str_item + ' '
        if result_str != '':
            result_str = result_str[:-2]
        result_similars.append(result_str)
        result_noun.append(noun)

    final_result = pd.DataFrame({' Noun': result_noun,
                                 'Adjectives and Verbs': result_similars})
    filename = 'Coals_clean_correlation.csv' 
    final_result.to_csv(filename, index=False)
    print('The resultant table has been saved as {}'.format(filename))



