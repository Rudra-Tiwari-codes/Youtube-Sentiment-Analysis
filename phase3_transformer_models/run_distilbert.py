"""Simple script runner to avoid import issues"""
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Now import and run
import sys
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    exec(open('05_distilbert_hf_trainer.py').read())
