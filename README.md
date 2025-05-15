# **NLP for clinical text classification**

Dataset: [https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions]
This dataset contains sample medical transcriptions for various medical specialties.

## Documents embeddings

This analysis processes and classifies clinical transcriptions by embedding text using a BERT-based model and visualizing them with t-SNE in 2D and 3D. 

![Embeddings](assets/embeddings.gif)

## Documents classification with SVM

Dataset has been balanced using SMOTE, and trained multiple classifiers (e.g., logistic regression, SVM) with hyperparameter tuning and GridSearchCV. The results are evaluated using classification reports and confusion matrices

<p align="center">
  <img src="assets/CM.png" alt="confusion matrix" width="900">
</p>

