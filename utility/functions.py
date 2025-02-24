import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, AutoConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import utility.data_preprocessing as dpp
import os
main_dir = "."


def set_model_name(model_selected):
    model_selected = model_selected.lower()
    if model_selected == 'distilbert':
        return 'distilbert-base-uncased'
    elif model_selected == 'bert':
        return 'bert-base-uncased'
    elif model_selected == 'roberta':
        return 'roberta-base'
    else:
        raise ValueError("Model not recognized")


def setting_tokenizer(finetune_dir, model_selected):
    model_name = set_model_name(model_selected)
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    
    if os.path.exists(finetune_dir):
        tokenizer_ft = AutoTokenizer.from_pretrained(finetune_dir)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer_ft)
    else:
        tokenizer_ft = None
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return tokenizer, tokenizer_ft, data_collator


def finetune_dir_path(exp_name, model_selected):
    name = "finetuned_" + str(model_selected) + "_" + str(exp_name)
    finetune_dir = os.path.join(main_dir, "Models", name)
    return finetune_dir


def load_data(data_path, cluster_data, column_input = "text", column_answer = "labels", label_exist=False, test_data_labeled=True, train=False, train_labels=None):
    df_ref_cluster = dpp.make_dataset(data_path, cluster_data)

    return dpp.data_preprocessing(df_ref_cluster, column_input, column_answer, label_exist, test_data_labeled, train, train_labels)


def tokenizer_function(df, model_selected, finetuned=False, exp_name=None):
    model_name = set_model_name(model_selected)
    print(f"\nTokenizing... finetuned?: {finetuned} (model: {model_selected} - using '{model_name}')")
    finetune_dir = finetune_dir_path(exp_name, model_selected)
    tokenizer, tokenizer_ft, _ = setting_tokenizer(finetune_dir, model_selected)

    if(finetuned):
        tokenized = tokenizer_ft(df['text'], padding=True, truncation=True)
    else:
        tokenized = tokenizer(df['text'], padding=True, truncation=True)
    tokenized['labels'] = df['labels']
    
    return tokenized


def save_finetuned(model, tokenizer, exp_name, model_selected):
    finetune_dir = finetune_dir_path(exp_name, model_selected)
    model.save_pretrained(finetune_dir)
    tokenizer.save_pretrained(finetune_dir)


def create_model(count_label, model_selected):
    model_name = set_model_name(model_selected)
    config = AutoConfig.from_pretrained(model_name)
    config.hidden_dropout_prob = 0.4 # dropout for preventing overfitting
    config.num_labels = count_label

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, #"distilbert-base-uncased",
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

def generate_embeddings_bert(batch, model_selected, finetuned=True, count_label=7, exp_name=None):
    model_name = set_model_name(model_selected)
    finetune_dir = finetune_dir_path(exp_name, model_selected)
    _, _, data_collator = setting_tokenizer(finetune_dir, model_selected)

    if(finetuned):
        model = AutoModelForSequenceClassification.from_pretrained(finetune_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = count_label)

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


from scipy.stats import entropy
def cluster_entropy(clusters):
    total_entropy = 0
    total_points = sum(len(cluster) for cluster in clusters)

    for cluster in clusters:
        prob_dist = np.bincount(cluster) / len(cluster) 
        cluster_ent = entropy(prob_dist)
        total_entropy += (len(cluster) / total_points) * cluster_ent

    return total_entropy


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
        results.append([original_labels[i], true, false, round(correctness, 2), round(error, 2)])

    cm_true = sum(row[i] for i, row in enumerate(conf_matrix))
    cm_total = conf_matrix.sum()
    cm_false = cm_total - cm_true
    results.append(["Total", cm_true, cm_false, round(cm_true / cm_total, 2), round(cm_false / cm_total, 2)])

    columns = ['Label', 'Correct', 'Wrong', 'Correctness', 'Error']
    df_results = pd.DataFrame(results, columns=columns)

    return df_results


from sklearn.metrics import normalized_mutual_info_score, silhouette_score

def clustering_performance(df, embeddings, test_data_labeled):

    np_clusters = df['cluster'].values
    unique_clusters = np.unique(np_clusters)
    clusters = [np.where(np_clusters == cluster)[0] for cluster in unique_clusters]
    entropy_score = cluster_entropy(clusters)
    
    if len(df['cluster'].unique()) > 1:
        silhouette_scr = silhouette_score(embeddings, df['cluster'])
    else:
        silhouette_scr = None
        print("Silhouette score skipped: only one cluster detected.")

    if(test_data_labeled == 'y'):

        nmi_scr = normalized_mutual_info_score(df['labels'], df['cluster'])
        # if len(df['cluster'].unique()) > 1:
        #     silhouette_scr = silhouette_score(df[['labels', 'cluster']], df['cluster'])
        # else:
        #     silhouette_scr = None
        #     print("Silhouette score skipped: only one cluster detected.")

        np_labels = df['labels'].values
        purity = calculate_purity(clusters, np_labels)
        
    else:
        nmi_scr = 0
        purity = 0

    return nmi_scr, silhouette_scr, purity, entropy_score


def generate_embeddings(embedding_method, processed_df_cluster, count_label, model_selected, exp_name=None):
    if embedding_method == 1:
        embeddings = embeddings_tfidf(processed_df_cluster['text'])
        str_embedding_method = "TF-IDF"
        return embeddings, str_embedding_method
    elif embedding_method == 2:
        embeddings = embeddings_sentence_transformers(processed_df_cluster['text'])
        str_embedding_method = "Sentence Transformers"
        return embeddings, str_embedding_method
    elif embedding_method == 3:
        embeddings = embeddings_bert_wo_finetune(processed_df_cluster, count_label, model_selected)
        str_embedding_method = str(model_selected) + " (without fine-tuning)"
        return embeddings, str_embedding_method
    elif embedding_method == 4:
        embeddings = embeddings_bert_w_finetune(processed_df_cluster, count_label, exp_name, model_selected)
        str_embedding_method = str(model_selected) + " (with fine-tuning)"
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

def embeddings_bert_wo_finetune(txt_data, num_label, model_selected):
    dataset_cluster = Dataset.from_pandas(txt_data)
    tokenized_dataset_cluster = dataset_cluster.map(tokenizer_function, batched=True, fn_kwargs={'model_selected': model_selected, 'finetuned': False})
    embedded_dataset = tokenized_dataset_cluster.map(generate_embeddings_bert, batched=True, batch_size=16, fn_kwargs={'model_selected': model_selected, 'finetuned': False, 'count_label': num_label})
    embeddings_array = np.array(embedded_dataset['embeddings'], dtype=np.float32)
    embeddings = embeddings_array[:, 0, :]

    return embeddings

def embeddings_bert_w_finetune(txt_data, num_label, exp_name, model_selected):
    dataset_cluster = Dataset.from_pandas(txt_data)
    tokenized_dataset_cluster = dataset_cluster.map(tokenizer_function, batched=True, fn_kwargs={'model_selected': model_selected, 'finetuned': True, 'exp_name': exp_name})
    embedded_dataset = tokenized_dataset_cluster.map(generate_embeddings_bert, batched=True, batch_size=16, fn_kwargs={'model_selected': model_selected, 'finetuned': True, 'count_label': num_label, 'exp_name': exp_name})
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

def save_results(new_row):
    file_name = "transformers/compare_results.xlsx"
    result_columns = [
        "Experiment Name", "Model", "Train Dataset", "Train Rows", "Validation Dataset", "Validation Rows", 
        "Test Dataset", "Test Rows", "Number of Labels from Training Data", "Labels Used", 
        "Learning Rate", "Batch Size", "Number of Epochs", "Execution Duration", 
        "K-fold[k]", "Loss", "Accuracy", "Precision", "Recall", "F1-Score"
    ]
    if os.path.exists(file_name):
        df_result = pd.read_excel(file_name, index_col=None)
    else:
        data = {col: [None] for col in result_columns}
        df_result = pd.DataFrame(data)
        df_result.to_excel(file_name, index=False)

    df_result = pd.concat([df_result, pd.DataFrame([new_row])], ignore_index=True)
    df_result.to_excel(file_name, index=False)
