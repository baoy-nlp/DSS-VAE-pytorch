import sys

sys.path.append(".")
from metrics.bleu_scorer import BleuScoreEvaluator

pred_file = sys.argv[1]
gold_file = sys.argv[2]

print(BleuScoreEvaluator.evaluate_file(pred_file=pred_file, gold_files=gold_file))
