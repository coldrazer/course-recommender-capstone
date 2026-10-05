# Personalized Online Course Recommender System

Machine learning capstone: built and evaluated six recommender approaches on 233,306 ratings from 33,901 learners across a 307 course catalog.

## Headline results

* Best content based method (clustering learner segments): **66.3% hit rate at 10**, beating the most popular baseline (57.4%) by 9 points
* Best collaborative filtering (neural network embeddings): **test RMSE 0.106**, cutting the mean baseline error in half
* Embeddings reused as features let a random forest flag **90% of audits** in a 95/5 imbalanced classification task where accuracy alone is meaningless

## Approach

Two families, compared on the same held out test:

**Content based (unsupervised)**
* User profile: rating weighted genre vectors, dot product scoring
* Course similarity: bag of words on titles and descriptions, cosine similarity
* Clustering: PCA (9 components, 93% variance) + K means (k = 9), recommend what is popular within each learner segment

**Collaborative filtering (supervised)**
* Item based KNN (k = 10, MSD similarity)
* NMF (32 latent factors, 50 epochs SGD)
* Neural network: 16 dim learner and course embeddings, dot product + sigmoid, early stopping at 27 epochs

Evaluation protocol: hide one enrolled course per learner, check whether it appears in each method's top 10.

## What did not work

Reported honestly in the full report: the user profile method gives 60% of learners zero recommendations (fixed thresholds favor heavy users), and near duplicate courses in the catalog inflate similarity scores. Next steps: top N lists instead of thresholds, catalog dedup, and a hybrid of cluster popularity and embeddings.

## Contents

* `course_recommender_capstone.ipynb` — the full analysis, end to end
* `report.pdf` — the complete writeup with figures and evaluation tables

## Reproduce

```bash
pip install -r requirements.txt
jupyter notebook course_recommender_capstone.ipynb
```

Data: IBM ML321EN capstone datasets (ratings, course genres, course text, user profiles, test learners). Random seed 42 throughout; 80/20 train/test split for collaborative filtering.

## Author

Christopher Guerrero, October 2026. IBM Machine Learning Professional Certificate capstone.
