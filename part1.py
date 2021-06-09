import pandas as pd
import pickle
from nltk.tag.perceptron import PerceptronTagger


def identify(corpus_filename):
    # reading corpus
    print('Reading corpus ...')
    data = pd.read_csv(corpus_filename)['Review']
    pretrain = PerceptronTagger()  # pre-trained tag model
    print('\t There are {} sentences in corpus'.format(data.shape[0]))

    # analysis
    print('Identifying nouns, adjectives and verbs ...')
    nouns = []  # noun
    noun_expressions = ['NN', 'NNS', 'NNP', 'NNPS']
    adjectives_vervs = []  # adjective and verb
    adjective_verb_expressions = ['JJ', 'JJR', 'JJS', 'VB', 'VBZ', 'VBP', 'VBD', 'VBN', 'VBG']
    for line in data:
        result = pretrain.tag(line.lower().split())
        for word, tag in result:
            if tag in noun_expressions:
                nouns.append(word)
            elif tag in adjective_verb_expressions:
                adjectives_vervs.append(word)

    # removing repeat
    nouns = list(set(nouns))
    adjectives_vervs = list(set(adjectives_vervs))
    print('\tThere are {} nouns and {} adjectives + verbs'.format(len(nouns), len(adjectives_vervs)))

    # saving results
    with open('Identifying_result.bin', 'wb') as f:
        f.write(pickle.dumps([nouns, adjectives_vervs]))
    print('The Identifying results are saved into "Identifying_result.bin".')
