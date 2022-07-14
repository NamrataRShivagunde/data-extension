import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("datasetpath", default=None, type=str)

args = parser.parse_args()

os.system('python evaluation.py {} bert-base-uncased'.format(args.datasetpath))
os.system('python evaluation.py {} bert-large-uncased'.format(args.datasetpath))

os.system('python evaluation.py {} distilbert-base-uncased'.format(args.datasetpath))

os.system('python evaluation.py {} roberta-base'.format(args.datasetpath))
os.system('python evaluation.py {} roberta-large'.format(args.datasetpath))

os.system('python evaluation.py {} albert-base-v1'.format(args.datasetpath))
os.system('python evaluation.py {} albert-large-v1'.format(args.datasetpath))
os.system('python evaluation.py {} albert-xlarge-v1'.format(args.datasetpath))
os.system('python evaluation.py {} albert-xxlarge-v1'.format(args.datasetpath))

os.system('python evaluation.py {} albert-base-v2'.format(args.datasetpath))
os.system('python evaluation.py {} albert-large-v2'.format(args.datasetpath))
os.system('python evaluation.py {} albert-xlarge-v2'.format(args.datasetpath))
os.system('python evaluation.py {} albert-xxlarge-v2'.format(args.datasetpath))

os.system('python evaluation.py {} t5-small'.format(args.datasetpath))
os.system('python evaluation.py {} t5-base'.format(args.datasetpath))
os.system('python evaluation.py {} t5-large'.format(args.datasetpath))
os.system('python evaluation.py {} t5-3b'.format(args.datasetpath))

os.system('python evaluation.py {} gpt2'.format(args.datasetpath))
os.system('python evaluation.py {} gpt2-medium'.format(args.datasetpath))
os.system('python evaluation.py {} gpt2-large'.format(args.datasetpath))
# os.system('python evaluation.py {} EleutherAI/gpt-neo-1.3B'.format(args.datasetpath))

os.system('python evaluation.py {} gpt2-xl'.format(args.datasetpath))


