import pandas as pd
import numpy as np
import numpy.matlib as mat
import pickle


def initial_co_table(filename):
    dat = pd.read_csv(filename)['Review']
    sentences = ''
    for sentence in dat:
        sentences += sentence.lower() + ' '

    words_line = sentences.split()
    words = list(set(words_line))
    words.sort()
    word_indices = {}
    for word in words:
        word_indices[word] = words.index(word)
    dimension = len(words)
    matrix = np.zeros((dimension, dimension), np.float32)

    for i in range(0, len(words_line) - 1):
        try:
            matrix[word_indices[words_line[i]], word_indices[words_line[i + 1]]] += 4
            matrix[word_indices[words_line[i + 1]], word_indices[words_line[i]]] += 4
            matrix[word_indices[words_line[i]], word_indices[words_line[i + 2]]] += 3
            matrix[word_indices[words_line[i + 2]], word_indices[words_line[i]]] += 3
            matrix[word_indices[words_line[i]], word_indices[words_line[i + 3]]] += 2
            matrix[word_indices[words_line[i + 3]], word_indices[words_line[i]]] += 2
            matrix[word_indices[words_line[i]], word_indices[words_line[i + 4]]] += 1
            matrix[word_indices[words_line[i + 4]], word_indices[words_line[i]]] += 1
        except IndexError:
            continue
    return words, matrix


def correlation_normalization(w):
    T = np.sum(w)
    dimension = w.shape[0]
    wa = mat.repmat(np.sum(w, axis=1), dimension, 1).transpose()
    wb = mat.repmat(np.sum(w, axis=0), dimension, 1)
    matrix = (T * w - wa * wb) / np.sqrt(wa * (T - wa) * wb * (T - wb))
    return matrix


def filter_negative(w):
    w[np.where(w < 0)] = 0
    return np.sqrt(w)


def COALS(corpus_filename, k):
    print('Starting COALS matrix ...')
    word_list, table5_matrix = initial_co_table(corpus_filename)
    print('Initial co-occurrence table with a rampled, 4-word window has been made.')
    table6_matrix = correlation_normalization(table5_matrix)
    print('Raw counts are converted to correlations.')
    table7_matrix = filter_negative(table6_matrix)
    print('Negative values are discarded and the positive values are square rooted.')

    # SVD
    print('performing SVD decomposition now ...')
    u, s, v = np.linalg.svd(table7_matrix, full_matrices=1, compute_uv=1)
    # table7_matrix = np.matmul(np.matmul(u, np.diag(s)), v)
    # table7_matrix = np.matmul(np.matmul(u[:, :k], np.diag(s[:k])), v[:k, :])
    print('\tSVD decomposition is finished.')
    table7_matrix = np.matmul(np.matmul(table7_matrix, v[:k, :].transpose()), np.linalg.inv(np.diag(s[:k])))
    print('\t New matrix with {} columns is produced based on SVD decomposition.'.format(k))
    # saving results
    with open('COALS_data.bin', 'wb') as f:
        f.write(pickle.dumps([word_list, table7_matrix]))
    print('The COALS vectors are saved into "COALS_data.bin".')
