import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset

import utility.data_preprocessing as dpp
import utility.functions as fc

from sklearn.metrics import confusion_matrix
from sklearn.cluster import SpectralClustering, AffinityPropagation, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from sklearn.preprocessing import LabelEncoder

main_dir = "."

# data_filename = "Consensus_Data_One_Label_EK11-24.xlsx"

data_filename = "Train_test Splits for FastFit (100% agreement).xlsx"
# data_filename = "Train_test Splits for FastFit (_0.8 Krippendorff).xlsx"

data = os.path.join(main_dir, "Data", data_filename)

sheet_names = ['D-ESP4-1','D-ESU4-1','D-ESP4-2','D-ESU4-2','D-ESP4-3','D-ESU4-3','D-ESP4-4','D-ESU4-4']
# train_sheet_names = ['D-ESP4-1','D-ESP4-2','D-ESP4-3','D-ESP4-4']
# test_sheet_names  = ['D-ESU4-1','D-ESU4-2','D-ESU4-3','D-ESU4-4']
train_sheet_names = ['Train Split']
test_sheet_names  = ['Test Split']


# 1. Load data
def load_data(cluster_data):
    df_ref_cluster = dpp.make_dataset(data, cluster_data)

    column_input = "text"
    column_answer = "labels"

    return dpp.data_preprocessing(df_ref_cluster, column_input, column_answer)

# 2. Embeddings
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
    tokenized_dataset_cluster = dataset_cluster.map(fc.tokenizer_function, batched=True, fn_kwargs={'finetuned': False})
    embedded_dataset = tokenized_dataset_cluster.map(fc.generate_embeddings, batched=True, batch_size=16, fn_kwargs={'finetuned': False, 'count_label': num_label})
    embeddings_array = np.array(embedded_dataset['embeddings'], dtype=np.float32)
    embeddings = embeddings_array[:, 0, :]

    return embeddings

def embeddings_bert_w_finetune(txt_data, num_label):
    dataset_cluster = Dataset.from_pandas(txt_data)
    tokenized_dataset_cluster = dataset_cluster.map(fc.tokenizer_function, batched=True, fn_kwargs={'finetuned': True})
    embedded_dataset = tokenized_dataset_cluster.map(fc.generate_embeddings, batched=True, batch_size=16, fn_kwargs={'finetuned': True, 'count_label': num_label})
    embeddings_array = np.array(embedded_dataset['embeddings'], dtype=np.float32)
    embeddings = embeddings_array[:, 0, :]

    return embeddings


# 3. Clustering
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

# 4. Output
def print_cluster_output(df):
    pd.set_option('display.max_colwidth', 90)
    pd.set_option('display.width', 150)

    for cluster, group in df.groupby('cluster'):
        print(f"Cluster {cluster}:")
        print(group[['text', 'original_labels', 'cluster']])
        print("\n")

# 5. Evaluation
def clustering_evaluation(df, str_clustering_method, str_embedding_method, str_reflection):
    # 5-1. Confusion matrix
    label_encoder = LabelEncoder()
    cluster_to_label = (
        df.groupby('cluster')['original_labels']
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )

    df['predicted_label'] = df['cluster'].map(cluster_to_label)
    true_labels_encoded = label_encoder.fit_transform(df['original_labels'])
    pred_labels_encoded = label_encoder.transform(df['predicted_label'])

    conf_matrix_clst = confusion_matrix(true_labels_encoded, pred_labels_encoded)

    conf_matrix_df = pd.DataFrame(
        conf_matrix_clst,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", cbar=True)

    plt.xticks(fontsize=9, rotation=45, ha='right')
    plt.yticks(fontsize=9)
    plt.title(f"Confusion Matrix Heatmap - {str_clustering_method} with {str_embedding_method}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()

    savefig_name = "clustering/fig/" + str_clustering_method + " with " + str_embedding_method + "(" + str_reflection + ").png"

    plt.savefig(savefig_name, dpi=300, bbox_inches='tight')
    # plt.show()

    # 5-2. Correctness
    unique_labels = df[['original_labels', 'labels']].drop_duplicates()
    sorted_labels = unique_labels.sort_values(by='labels')
    original_labels = sorted_labels['original_labels'].tolist()
    df_correctness_clst = fc.correctness(conf_matrix_clst, original_labels)

    # 5-3. Clustering performance evaluation
    nmi_spectral, silhouette_spectral, purity = fc.clustering_performance(df)

    # user_notes = input("Leave notes for this experiment: \n")

    file_name = "clustering/result/"+ str_embedding_method + " using " + str_reflection + ".txt"
    with open(file_name, "a") as file:
        # file.write(f"\n**{user_notes}\n")
        file.write(f"<<<{str_clustering_method}>>>")
        file.write(f"\n# Weighted Averages of Error and Correctness of {str_clustering_method} with {str_embedding_method}\n")
        file.write(df_correctness_clst.to_string(index=False, header=True))

        file.write(f"\n# Evaluation of {str_clustering_method} with {str_embedding_method}")
        file.write(f"\nNMI: {nmi_spectral}")
        file.write(f"\nSilhouette Score: {silhouette_spectral}")
        file.write(f'\nPurity: {purity:.2f}')
        file.write("\n------------------------------\n")

# 8. Data visualization
# import umap

# umap_model = umap.UMAP(n_components=2, random_state=42)
# umap_results = umap_model.fit_transform(embeddings_cls)

# plt.figure(figsize=(6, 4))
# # plt.scatter(umap_results[:, 0], umap_results[:, 1], c=labels, cmap='Paired') 
# for i in range(count_label):
#     plt.scatter(umap_results[labels == i, 0], umap_results[labels == i, 1], label=f'{i}: {original_labels[i]}')
# plt.legend(title="Labels", fontsize=8)

# plt.colorbar()
# plt.title("UMAP Visualization of Spectral Clustering")
# plt.show()


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


def execute_clustering(embeddings, str_embedding_method, str_reflection, processed_df_cluster, count_label):
    str_clustering_method = "Spectral Clustering"
    clustered_data_sp = clustering_spectral(embeddings, processed_df_cluster, count_label)
    clustering_evaluation(clustered_data_sp, str_clustering_method, str_embedding_method, str_reflection)

    str_clustering_method = "Affinity Propagation Clustering"
    clustered_data_ap = clustering_affinity_propagation(embeddings, processed_df_cluster, count_label)
    clustering_evaluation(clustered_data_ap, str_clustering_method, str_embedding_method, str_reflection)

    str_clustering_method = "K-means Clustering"
    clustered_data_km = clustering_kmeans(embeddings, processed_df_cluster, count_label)
    clustering_evaluation(clustered_data_km, str_clustering_method, str_embedding_method, str_reflection)

    print(f"{str_embedding_method} for {str_reflection} done")


def execute_all(cluster_data, str_reflection):
    processed_df_cluster, count_label, class_weights_tensor = load_data(cluster_data)
    embeddings, str_embedding_method = generate_embeddings(embedding_method, processed_df_cluster, count_label)
    execute_clustering(embeddings, str_embedding_method, str_reflection, processed_df_cluster, count_label)

str_reflection = ""
str_embedding_method = ""
str_clustering_method = ""

while True:
    try:
        embedding_method = int(input(
            "\nChoose embedding methods: \n"
            "\t1: TF-IDF, \n"
            "\t2: Sentence Transformers, \n"
            "\t3: DistilBERT without finetuning, \n"
            "\t4: DistilBERT with finetuning, \n"
            "\t0: Exit \n"
        ))

        if embedding_method == 0:
            print("Exiting the program.")
            break
        elif embedding_method in (1, 2, 3, 4):
            print(f"You selected method {embedding_method}.")
            break
        else:
            print("Invalid input. Please choose a valid number.")
    except ValueError:
        print("Invalid input. Please enter a number.")


try:
    ref_num = input("\nSelect how to cluster the dataset (0: All reflections together, 1: Each reflection separately): ")
    int_ref_num = int(ref_num)
except ValueError:
    print("Invalid input. Please enter a number.")


if(int_ref_num == 0):
    cluster_data = test_sheet_names
    str_reflection = "Entire Reflection"
    execute_all(cluster_data, str_reflection)
else:
    for i in range(len(test_sheet_names)):
        cluster_data = [test_sheet_names[i]]
        str_reflection = "Reflection " + str(i + 1)
        execute_all(cluster_data, str_reflection)