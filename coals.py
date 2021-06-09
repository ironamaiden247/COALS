'''
Correlated Occurrence Analogue to Lexical Semantics (COALS)

Rohde, D.L., L.M. Gonnerman, and D.C. Plaut, An improved model of 
semantic similarity based on lexical co-occurrence. Communications of 
the ACM, 2006. 8(627-633): p. 116.

Reference codes for python implementation:
http://tedlab.mit.edu/~dr/COALS/
https://github.com/hbrouwer/coals
https://github.com/fozziethebeat/S-Space/wiki/Coals
'''

identified_noun_adjective_verb = False
if not identified_noun_adjective_verb:
    from part1 import identify
    identify(corpus_filename='DatasetSemanticSimilarity.csv')
    print()

made_COALS_matrix = False
if not made_COALS_matrix:
    from part2 import COALS
    COALS(corpus_filename='DatasetSemanticSimilarity.csv', k=50)
    print()

find_adjective_verb = True
if find_adjective_verb:
    from part3 import finding_corr_threshold
    finding_corr_threshold(corr_threshold=0.5)
    print()
  