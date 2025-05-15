import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # Importa supporto 3D
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier


def main():

    clinical_text_df = pd.read_csv("mtsamples.csv")

    print(clinical_text_df.columns)
    print(clinical_text_df.head(5))

    clinical_text_df = clinical_text_df[clinical_text_df['transcription'].notna()]
    sent_count,word_count= get_sentence_word_count(clinical_text_df['transcription'].tolist())
    print("Number of sentences in transcriptions column: "+ str(sent_count))
    print("Number of unique words in transcriptions column: "+str(word_count))


    data_categories  = clinical_text_df.groupby(clinical_text_df['medical_specialty'])
    i = 1
    print('===========Original Categories =======================')
    for catName,dataCategory in data_categories:
        print('Cat:'+str(i)+' '+catName + ' : '+ str(len(dataCategory)) )
        i = i+1
    print('======================================================')

    #Remove the categories with less than 250 samples
    filtered_data_categories = data_categories.filter(lambda x:x.shape[0] > 250)
    final_data_categories = filtered_data_categories.groupby(filtered_data_categories['medical_specialty'])
    i=1
    print('============Reduced Categories ======================')
    for catName,dataCategory in final_data_categories:
        print('Cat:'+str(i)+' '+catName + ' : '+ str(len(dataCategory)) )
        i = i+1

    print('============ Reduced Categories ======================')

    #Plot of selected categories
    plt.figure(figsize=(20,15))
    sns.countplot(y='medical_specialty', data = filtered_data_categories )
    plt.show()

    data = filtered_data_categories[['transcription', 'medical_specialty']]
    data = data.drop(data[data['transcription'].isna()].index)

    print(repr(data['medical_specialty'].unique()))

    data['medical_specialty'] = data['medical_specialty'].str.strip()

    data['medical_specialty'] = data['medical_specialty'].replace({
        'Consult - History and Phy.': 'General Medicine'
    })

    # Plot of selected categories
    plt.figure(figsize=(20, 15))
    sns.countplot(y='medical_specialty', data=data)
    plt.show()

    print(f"DEBUG: data shape: {data.shape}")

    print("====================TEXT CLEANING====================")
    print("ORIGINAL DATA SAMPLES")
    print('Sample Transcription 1:'+data.iloc[5]['transcription']+'\n')
    print('Sample Transcription 2:'+data.iloc[125]['transcription']+'\n')
    print('Sample Transcription 3:'+data.iloc[1000]['transcription'])
    print("=====================================================")

    #data['transcription'] = data['transcription'].apply(lemmatize_text) #Using BERT, lemmatize is not necessary
    # data['transcription'] = data['transcription'].apply(clean_text)
    #
    # print("====================TEXT CLEANING====================")
    # print("CLEANED DATA SAMPLES")
    # print('Sample Transcription 1:'+data.iloc[5]['transcription']+'\n')
    # print('Sample Transcription 2:'+data.iloc[125]['transcription']+'\n')
    # print('Sample Transcription 3:'+data.iloc[1000]['transcription'])
    # print("=====================================================")

    #Load of MiniLM encoder
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(data['transcription'].tolist(), show_progress_bar=True)

    #PLOT
    #ogni punto è una transcription
    #ogni colore è la specialità medica associata
    #le posizioni x,y ono coordinate generate da t-SNE, che cerca di preservare le relazioni di similarità tra i dati originariamente ad alta dimensione (quelli della matrice TF-IDF)


    # Estrai la terza dimensione
    tsne = TSNE(n_components=3, perplexity=40, n_iter=300, random_state=0)
    tsne_results = tsne.fit_transform(embeddings)

    # Aggiungi le colonne al DataFrame
    data['tsne_x'] = tsne_results[:,0]
    data['tsne_y'] = tsne_results[:,1]
    data['tsne_z'] = tsne_results[:, 2]

    # Plot 2D
    plt.figure(figsize=(12,10))
    sns.scatterplot(
        x='tsne_x', y='tsne_y',
        hue='medical_specialty',
        data=data,
        legend='full',
        alpha=0.8
    )
    plt.title("t-SNE projection of BERT embeddings")
    plt.show()

    # Plot 3D
    import plotly.express as px
    data["constant_size"] = 1
    fig = px.scatter_3d(
        data,
        x='tsne_x',
        y='tsne_y',
        z='tsne_z',
        color='medical_specialty',
        size="constant_size",  # dimensione dei punti
        size_max=10,
        opacity=0.9,
        hover_data=['medical_specialty'],
        template="plotly_white"
    )
    fig.update_layout(title="t-SNE 3D Projection of BERT Embeddings")
    fig.show()

    print("====MODEL TRAINING=====")

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        data['medical_specialty'],
        test_size=0.2,
        random_state=42, # mantieni distribuzione etichette
    )

    #Unbalanced dataset handling
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train,y_train)

    plt.figure(figsize=(20,15))
    sns.countplot(y=pd.Series(y_resampled))
    plt.title('Categories dimension after SMOTE')
    plt.show()

    print(f"DEBUG: X_resampled size: {len(X_resampled)}")
    print(f"DEBUG: y_resampled size: {len(y_resampled)}")

    model_types = ["logistic", "svm"]

    for model_type in model_types:
        train_model(X_resampled, y_resampled, X_test, y_test, model_type)

def get_sentence_word_count(text_list):
    sent_count = 0
    word_count = 0
    vocab = {}
    for text in text_list:
        sentences = sent_tokenize(str(text).lower())
        sent_count = sent_count + len(sentences)
        for sentence in sentences:
            words = word_tokenize(sentence)
            for word in words:
                if (word in vocab.keys()):
                    vocab[word] = vocab[word] + 1
                else:
                    vocab[word] = 1
    word_count = len(vocab.keys())
    return sent_count, word_count
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler  # utile per MLP e SVM

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_model(X_train, y_train, X_test, y_test, model_type="logistic"):
    print(f"\n=== Training model: {model_type.upper()} with GridSearchCV ===")

    if model_type == "logistic":
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000))
        ])

        param_grid = {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs', 'liblinear'],
            'clf__class_weight': [None, 'balanced']
        }

    elif model_type == "mlp":
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(max_iter=1000, random_state=42))
        ])

        param_grid = {
            'clf__hidden_layer_sizes': [(64,), (128,), (64, 64), (128, 64), (64, 128, 64), (8,), (16,), (8, 8), (16, 8), (8, 16, 8)],
            'clf__activation': ['relu', 'tanh'],
            'clf__alpha': [0.0001, 0.001, 0.01],
            'clf__learning_rate': ['constant', 'adaptive']
        }

    elif model_type == "svm":
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(probability=True))
        ])

        param_grid = {
            'clf__C': [0.1, 1, 10, 100],
            'clf__kernel': ['linear', 'rbf'],
            'clf__class_weight': [None, 'balanced']
        }

    elif model_type == "random_forest":
        pipe = Pipeline([
            ('clf', RandomForestClassifier(random_state=42))
        ])

        param_grid = {
            'clf__n_estimators': [100, 200, 500, 1000],
            'clf__max_depth': [None, 10, 20, 50],
            'clf__min_samples_split': [2, 5],
            'clf__min_samples_leaf': [1, 2],
            'clf__class_weight': [None, 'balanced']
        }

    elif model_type == "xgboost":
        pipe = Pipeline([
            ('clf', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
        ])

        param_grid = {
            'clf__n_estimators': [100, 200, 500, 1000],
            'clf__max_depth': [3, 6],
            'clf__learning_rate': [0.01, 0.1, 0.3],
            'clf__subsample': [0.8, 1.0],
            'clf__colsample_bytree': [0.8, 1.0]
        }

    else:
        raise ValueError(f"Model type '{model_type}' not supported.")

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    print("\n=== Best parameters ===")
    print(grid.best_params_)

    # Predict and evaluate
    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f",
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), cmap="Blues")
    plt.title(f"Confusion Matrix - {model_type.upper()}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


from itertools import combinations
from sklearn.metrics.pairwise import euclidean_distances


if __name__ == "__main__":
    main()





