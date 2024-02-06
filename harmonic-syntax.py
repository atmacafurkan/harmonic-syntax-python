# Using anytree in Python
from anytree import Node
import csv
from typing import List

# Custom node class with a named list field
class SyntaxNode(Node):
    def __init__(self, name, label = None, merge_feat = None, neutral_feats=None, agree_feats = None, domination_count = None, parent=None, children=None):
        super(SyntaxNode, self).__init__(name, parent, children)
        self.label = label if label is not None else {}
        self.merge_feat = merge_feat if merge_feat is not None else {}
        self.domination_count = domination_count if domination_count is not None else 0
        self.agree_feats = agree_feats if agree_feats is not None else {}
        self.neutral_feats = neutral_feats if neutral_feats is not None else {}

        # Additional checks for number of parents and children
        if parent is not None and len(parent) > 1: # check parent
            raise ValueError("SyntaxNode can have at most one parent.")
        
        if children is not None and len(children) > 2: # check child
            raise ValueError("SyntaxNode can have at most two children.")

def read_csv_to_nodes(csv_file_path: str) -> List[SyntaxNode]:
    def parse_feats(feat_str):
        return [feat.strip() for feat in feat_str.split('-')] if feat_str is not None else []

    nodes = []
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            node = SyntaxNode(
                name=row['name'],
                merge_feat=parse_feats(row['merge_feat']),
                agree_feats=parse_feats(row['agree_feats']),
                neutral_feats=parse_feats(row['neutral_feats']),
                label=parse_feats(row['label'])
            )
            nodes.append(node)
    return nodes