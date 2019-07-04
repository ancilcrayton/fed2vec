import os
import argparse
import pandas as pd
import multiprocessing
import swifter
from utils import text_preprocessing, tag_documents
from gensim.models.doc2vec import Doc2Vec

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', help='Number of epochs to train', default=25, type=int)
parser.add_argument('--vec_size', help='Size of embedding vector', default=300, type=int)
parser.add_argument('--window', help='Context window size for training', default=2, type=int)
parser.add_argument('--alpha', help='Learning rate', default=0.025)
parser.add_argument('--algorithm', help="Choose distributed memory ('DM') or Distributed Bag of Words ('DBOW')", default='DM')
args = parser.parse_args()

# Collect data
raw_data = pd.read_json('data/fed_speeches.json', orient='records')

# Preprocess speeches
print('Preprocessing {} speeches...'.format(len(raw_data.index))
processed_speeches = raw_data['Speech'].swifter.apply(lambda x: text_preprocessing(x)) # use swifter to parallelize the apply function
print('done preprocessing.')

# Tagging documents
print('Tagging documents...')
tagged_docs = tag_documents(processed_speeches)
print('done tagging.')

# Model building
print('Building model...')

# Set parameters
max_epochs = args.epochs
window = args.window
vec_size = args.vec_size
alpha = args.alpha
cpus = multiprocessing.cpu_count()

# Choose the algorithm for training
if args.algorithm == 'DM':
    model = Doc2Vec(size=vec_size, window=window, alpha=alpha, min_alpha=0.00025, min_count=1, dm =1, workers=cpus)
elif args.algorithm == 'DBOW':
    model = Doc2Vec(size=vec_size, window=window, alpha=alpha, min_alpha=0.00025, min_count=1, dm =0, workers=cpus)
else:
    raise ValueError("Choose either 'DM' or 'DBOW' for the algorithm!")

# Build vocabulary
model.build_vocab(tagged_docs)
print('model built.')

# Training the model
print('Training model...')
for epoch in range(max_epochs):
    print('Epoch: {}'.format(epoch))
    # Train model
    model.train(tagged_docs, total_examples = model.corpus_count, epochs = model.iter)
    # Decrease learning rate
    model.alpha -= 0.0002
    # Fix the learning rate, no decay
    model.min_alpha = model.alpha
print('model trained.')

# Saving model into results directory
print('Saving model into results...')

if os.path.isdir('results/') == False:
    os.system('mkdir results/')

model.save("results/fed2vec.model")
print('Model saved!')
