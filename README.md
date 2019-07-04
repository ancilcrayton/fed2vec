# Fed2Vec: Learning Distributed Representations of Speeches by the Federal Reserve

## Purpose
In this project, I present Fed2Vec, which is a Doc2Vec model trained on speeches by members of the Federal Reserve Board of Governors. I learn the word and document embeddings of Federal Reserve language to learn semantics and contextualize the language. There are many ways this model can be used to study interesting research questions related to central banks, monetary policy, and public policy in general. Some potential uses of this model could be:

1. Find closest speeches between different speakers to compare monetary policy stances.
2. Get embeddings calculations for speeches by other central banks and see how they compare against the Federal Reserve.
3. Compare distances between speeches by the Federal Reserve chairs/chairmen and different governors.
4. Use speeches to predict financial market fluctuations, macroeconomic outcomes, or future monetary policy stances

This project features webscraping, deep learning for NLP, dimensionality reduction, and visualization.

## Usage
This project was developed in Python 3.6 and dependecies should be installed by running the command `pip install -r requirements.txt`.

First, collect the speeches, which is scraped from the [Federal Reserve Board of Governors](https://www.federalreserve.gov/) website by running the `get_speeches.py` script:
```
$ python get_speeches.py
```

Second, perform text preprocessing, document tagging, and train the model by running the `train.py` script:
```
$ python train.py
```

The `train.py` script has five different arguments that can be passed in:

`--epochs`: The number of epochs for training (int, default=25)

`--vec_size`: The size of the embedding vectors (int, default=300)

`--window`: The context window size for the training of the word embeddings (int, default=2)

`--alpha`: The learning rate for gradient descent (float, default=0.025)

`--algorithm`: The algorithm for training, either distributed memory ('DM') or distributed bag of words ('DBOW') (string, default='DM')

An example of training with 10 epochs, embedding size of 20, a window of 5, learning rate of 0.2, and the distributed bag of words algorithm:

```
$ python train.py --epochs 10 --vec_size 20 --window 5 --alpha 0.2 --algorithm 'DBOW'
```

Finally, after the model is trained, the saved Fed2vec model will be saved in the `results/` directory as `fed2vec.model`.

To show simple examples of exploring the reuslts of the model, I include a notebook in the `notebooks/` directory named `visualize_embeddings.ipynb`.

