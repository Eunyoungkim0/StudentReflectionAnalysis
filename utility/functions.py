import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, AutoModelForSequenceClassification, AutoConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import utility.data_preprocessing as dpp
import os
main_dir = "."
finetune_dir = os.path.join(main_dir, "finetuned_distilbert")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", clean_up_tokenization_spaces=True)
if os.path.exists(finetune_dir):
    tokenizer_ft = AutoTokenizer.from_pretrained(finetune_dir)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def load_data(data_path, cluster_data, column_input = "text", column_answer = "labels", label_exist=False):
    df_ref_cluster = dpp.make_dataset(data_path, cluster_data)

    return dpp.data_preprocessing(df_ref_cluster, column_input, column_answer, label_exist)


def tokenizer_function(df, finetuned=False):
    if(finetuned):
        tokenized = tokenizer_ft(df['text'], padding=True, truncation=True)
    else:
        tokenized = tokenizer(df['text'], padding=True, truncation=True)
    tokenized['labels'] = df['labels']
    
    return tokenized

def save_finetuned(model, tokenizer):
    model.save_pretrained(finetune_dir)
    tokenizer.save_pretrained(finetune_dir)

def create_model(count_label):
    config = AutoConfig.from_pretrained("distilbert-base-uncased")
    config.hidden_dropout_prob = 0.4 # dropout for preventing overfitting
    config.num_labels = count_label

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        config=config
    )

    return model

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=1)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def generate_embeddings_bert(batch, finetuned=True, count_label=7):
    if(finetuned):
        model = AutoModelForSequenceClassification.from_pretrained(finetune_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = count_label)
        
    inputs = data_collator({'input_ids' : batch['input_ids']})
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    embeddings = torch.mean(torch.stack(hidden_states[-6:]), dim=0)
    return {'embeddings' : embeddings.detach().cpu()}


from collections import Counter

def calculate_purity(clusters, labels):
    total_points = 0
    total_purity = 0

    for cluster in clusters:
        total_points += len(cluster)
        cluster_labels = labels[cluster]
        most_common_label_count = Counter(cluster_labels).most_common(1)[0][1]
        total_purity += most_common_label_count

    purity = total_purity / total_points
    return purity


def correctness(conf_matrix, original_labels):
    results = []
    cm_true = 0
    cm_false = 0
    cm_total = 0

    for i, row in enumerate(conf_matrix):
        row = np.array(row, dtype=int)
        true = row[i]
        total = row.sum()
        false = total - true
        correctness = true / total
        error = false / total
        results.append([original_labels[i], true, false, correctness, error])

    cm_true = sum(row[i] for i, row in enumerate(conf_matrix))
    cm_total = conf_matrix.sum()
    cm_false = cm_total - cm_true
    results.append(["Total", cm_true, cm_false, cm_true / cm_total, cm_false / cm_total])

    columns = ['Label', 'Correct', 'Wrong', 'Correctness', 'Error']
    df_results = pd.DataFrame(results, columns=columns)

    return df_results


from sklearn.metrics import normalized_mutual_info_score, silhouette_score

def clustering_performance(df):

    nmi_spectral = normalized_mutual_info_score(df['labels'], df['cluster'])
    # silhouette_spectral = silhouette_score(df[['labels', 'cluster']], df['cluster'])

    if len(df['cluster'].unique()) > 1:
        silhouette_spectral = silhouette_score(df[['labels', 'cluster']], df['cluster'])
    else:
        silhouette_spectral = None
        print("Silhouette score skipped: only one cluster detected.")

    np_clusters = df['cluster'].values
    np_labels = df['labels'].values
    unique_clusters = np.unique(np_clusters)
    clusters = [np.where(np_clusters == cluster)[0] for cluster in unique_clusters]

    purity = calculate_purity(clusters, np_labels)

    return nmi_spectral, silhouette_spectral, purity


def generate_embeddings(embedding_method, processed_df_cluster, count_label):
    if embedding_method == 1:
        embeddings = embeddings_tfidf(processed_df_cluster['text'])
        str_embedding_method = "TF-IDF"
        return embeddings, str_embedding_method
    elif embedding_method == 2:
        embeddings = embeddings_sentence_transformers(processed_df_cluster['text'])
        str_embedding_method = "Sentence Transformers"
        return embeddings, str_embedding_method
    elif embedding_method == 3:
        embeddings = embeddings_bert_wo_finetune(processed_df_cluster, count_label)
        str_embedding_method = "DistilBERT (without fine-tuning)"
        return embeddings, str_embedding_method
    elif embedding_method == 4:
        embeddings = embeddings_bert_w_finetune(processed_df_cluster, count_label)
        str_embedding_method = "DistilBERT (with fine-tuning)"
        return embeddings, str_embedding_method

### ----------------------------------------------
### Different Embedding Methods for Clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from datasets import Dataset

def embeddings_tfidf(txt_data):
    tfidf_vectorizer = TfidfVectorizer()
    embeddings = tfidf_vectorizer.fit_transform(txt_data)

    return embeddings

def embeddings_sentence_transformers(txt_data):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(txt_data.tolist())

    return embeddings

def embeddings_bert_wo_finetune(txt_data, num_label):
    dataset_cluster = Dataset.from_pandas(txt_data)
    tokenized_dataset_cluster = dataset_cluster.map(tokenizer_function, batched=True, fn_kwargs={'finetuned': False})
    embedded_dataset = tokenized_dataset_cluster.map(generate_embeddings_bert, batched=True, batch_size=16, fn_kwargs={'finetuned': False, 'count_label': num_label})
    embeddings_array = np.array(embedded_dataset['embeddings'], dtype=np.float32)
    embeddings = embeddings_array[:, 0, :]

    return embeddings

def embeddings_bert_w_finetune(txt_data, num_label):
    dataset_cluster = Dataset.from_pandas(txt_data)
    tokenized_dataset_cluster = dataset_cluster.map(tokenizer_function, batched=True, fn_kwargs={'finetuned': True})
    embedded_dataset = tokenized_dataset_cluster.map(generate_embeddings_bert, batched=True, batch_size=16, fn_kwargs={'finetuned': True, 'count_label': num_label})
    embeddings_array = np.array(embedded_dataset['embeddings'], dtype=np.float32)
    embeddings = embeddings_array[:, 0, :]

    return embeddings

### ----------------------------------------------
### Different Clustering Methods
from sklearn.cluster import SpectralClustering, AffinityPropagation, KMeans
from sklearn.metrics.pairwise import cosine_similarity

def clustering_spectral(embeddings, df, count_label):
    spectral = SpectralClustering(n_clusters=count_label, affinity='cosine') # Using cosine similarity
    labels = spectral.fit_predict(embeddings)
    df['cluster'] = labels

    return df

def clustering_affinity_propagation(embeddings, df, count_label):
    affinity_propagation = AffinityPropagation(affinity='precomputed', damping=0.9)
    similarity_matrix = cosine_similarity(embeddings)
    labels = affinity_propagation.fit_predict(similarity_matrix)
    df['cluster'] = labels

    return df

def clustering_kmeans(embeddings, df, count_label):
    kmeans = KMeans(n_clusters=count_label, random_state=42, n_init=15)
    kmeans.fit(embeddings)
    df['cluster'] = kmeans.labels_

    return df
### ----------------------------------------------