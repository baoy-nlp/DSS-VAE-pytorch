from __future__ import print_function

from nltk.tree import Tree


def load_trees(path, strip_top=True, strip_spmrl_features=True):
    trees = []
    with open(path) as infile:
        for line in infile:
            trees.append(Tree.fromstring(line))

    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label() in ("TOP", "ROOT"):
                assert len(tree) == 1
                trees[i] = tree[0]
    return trees
