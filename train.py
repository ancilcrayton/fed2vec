import argparse
import pandas as pd
from utils import text_preprocessing, tag_documents
from gensim.models.doc2vec import Doc2Vec

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', help='Number of epochs to train', default=25, type=int)
parser.add_argument('--vec-size', help='Size of embedding vector', default=300, type=int)
parser.add_argument('--alpha', help='Learning rate', default=0.025)
parser.add_arguement('--algorithm', help="Choose distributed memory ('DM') or Distributed Bag of Words ('DBOW')", default='DM')
args = parser.parse_args()


