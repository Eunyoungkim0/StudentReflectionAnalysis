import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import utility.functions as fc
import labeling_outputs_bertopic as label_topic

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

main_dir = "."

# data_filename = "Train_test Splits for FastFit (100% agreement).xlsx"
data_filename = "Train_test Splits for FastFit (_0.8 Krippendorff).xlsx"
data_path = os.path.join(main_dir, "Data", data_filename)

sheet_names = ['D-ESP4-1','D-ESU4-1','D-ESP4-2','D-ESU4-2','D-ESP4-3','D-ESU4-3','D-ESP4-4','D-ESU4-4']
train_sheet_names = ['Train Split']
test_sheet_names  = ['Test Split']


def save_cluster_output_to_excel(df, clustering_methods, str_embedding_method, mode="a"):
    output_file = f"clustering/result/cluster_output_{str_embedding_method}.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl", mode=mode) as writer:
        df[['original_text', 'original_labels', 'cluster', 'predicted_label', 'predicted_topics']].to_excel(writer, sheet_name=clustering_methods, index=True)


# 5. Evaluation
def clustering_evaluation(embeddings, df, str_clustering_method, str_embedding_method, str_reflection, mode="a"):
    # 5-1. Confusion matrix
    label_encoder = LabelEncoder()
    cluster_to_label = (df.groupby('cluster')['original_labels'].agg(lambda x: x.value_counts().idxmax()).to_dict())

    df['predicted_label'] = df['cluster'].map(cluster_to_label)
    df['predicted_label'] = df['predicted_label'].fillna('None')

    # 5-2. Label Clustering Outputs
    df = label_topic.extract_topic(df, embeddings)

    save_cluster_output_to_excel(df, str_clustering_method, str_embedding_method, mode)

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

    # 5-3. Correctness
    unique_labels = df[['original_labels', 'labels']].drop_duplicates()
    sorted_labels = unique_labels.sort_values(by='labels')
    original_labels = sorted_labels['original_labels'].tolist()
    df_correctness_clst = fc.correctness(conf_matrix_clst, original_labels)

    # 5-4. Clustering performance evaluation
    nmi_spectral, silhouette_spectral, purity = fc.clustering_performance(df)

    file_name = "clustering/result/"+ str_embedding_method + " using " + str_reflection + ".txt"
    with open(file_name, "a") as file:
        file.write(f"<<<{str_clustering_method}>>>")
        file.write(f"\n# Weighted Averages of Error and Correctness of {str_clustering_method} with {str_embedding_method}\n")
        file.write(df_correctness_clst.to_string(index=False, header=True))

        file.write(f"\n# Evaluation of {str_clustering_method} with {str_embedding_method}")
        file.write(f"\nNMI: {nmi_spectral}")
        file.write(f"\nSilhouette Score: {silhouette_spectral}")
        file.write(f'\nPurity: {purity:.2f}')
        file.write("\n------------------------------\n")


def execute_clustering(embeddings, str_embedding_method, str_reflection, processed_df_cluster, count_label):
    str_clustering_method = "Spectral Clustering"
    clustered_data_sp = fc.clustering_spectral(embeddings, processed_df_cluster, count_label)
    clustering_evaluation(embeddings, clustered_data_sp, str_clustering_method, str_embedding_method, str_reflection, "w")

    str_clustering_method = "Affinity Propagation Clustering"
    clustered_data_ap = fc.clustering_affinity_propagation(embeddings, processed_df_cluster, count_label)
    clustering_evaluation(embeddings, clustered_data_ap, str_clustering_method, str_embedding_method, str_reflection)

    str_clustering_method = "K-means Clustering"
    clustered_data_km = fc.clustering_kmeans(embeddings, processed_df_cluster, count_label)
    clustering_evaluation(embeddings, clustered_data_km, str_clustering_method, str_embedding_method, str_reflection)

    print(f"{str_embedding_method} for {str_reflection} done")


def execute_all(cluster_data, str_reflection):
    processed_df_cluster, count_label, class_weights_tensor = fc.load_data(data_path, cluster_data)
    processed_df_cluster['original_labels'] = processed_df_cluster['original_labels'].fillna('None')
    embeddings, str_embedding_method = fc.generate_embeddings(embedding_method, processed_df_cluster, count_label)
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