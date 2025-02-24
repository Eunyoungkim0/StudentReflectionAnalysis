#----------------------------------------------------------------------------------------
# Setting up Some Variables
str_reflection = ""
str_embedding_method = ""
str_clustering_method = ""

test_data_labeled = 'y'
exp_name = "r1_100"
int_ref_num = 0

# model_selected = 'DistilBERT'
model_selected = 'BERT'
# model_selected = 'RoBERTa'
#----------------------------------------------------------------------------------------
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
# data_filename = "Train_test Splits for FastFit (_0.8 Krippendorff).xlsx"
# data_filename = "Spring2025_reflection.xlsx"
# data_path = os.path.join(main_dir, "Data", data_filename)

# data_filename = "predictions_" + str(exp_name) + ".xlsx"
# data_path = os.path.join(main_dir, "transformers/result", data_filename)

if(exp_name == "all_100"):
    data_filename = "SL-R-ALL-100-2_11.xlsx"
    sheet_names = ['fixed_sl-all-100']
elif(exp_name == "all_80"):
    data_filename = "SL-R-ALL-80-2_11.xlsx"
    sheet_names = ['fixed_sl-all-80']
elif(exp_name == "r1_100"):
    data_filename = "SL-R1-100-2_11.xlsx"
    sheet_names = ['fixed_sl-r1-100']
elif(exp_name == "r1_80"):
    data_filename = "SL-R1-80-2_11.xlsx"
    sheet_names = ['fixed_sl-r1-80']

# sheet_names = ['Test_Data_Prediction']
data_path = os.path.join(main_dir, "Data", data_filename)

def save_cluster_output_to_excel(df, clustering_methods, str_embedding_method, mode="a"):
    output_file = f"clustering/result/cluster_output_{str_embedding_method}_{exp_name}.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl", mode=mode) as writer:
        if test_data_labeled == 'y':
            df[['original_text', 'original_labels', 'cluster', 'predicted_label', 'predicted_topics']].to_excel(writer, sheet_name=clustering_methods, index=True)
        else:
            column_name_transformer = 'predicted_label_' + model_selected
            df[column_name_transformer] = df[column_name_transformer].fillna('None')
            df[['original_text', 'original_labels', column_name_transformer, 'cluster', 'predicted_label', 'predicted_topics']].to_excel(writer, sheet_name=clustering_methods, index=True)


# 5. Evaluation
def clustering_evaluation(embeddings, df, str_clustering_method, str_embedding_method, str_reflection, count_label, mode="a"):
    print(f"\n# [{str_embedding_method} with {str_clustering_method}] ({exp_name})")
    # 5-1. Confusion matrix
    label_encoder = LabelEncoder()
    cluster_to_label = (df.groupby('cluster')['original_labels'].agg(lambda x: x.value_counts().idxmax()).to_dict())

    df['predicted_label'] = df['cluster'].map(cluster_to_label)
    df['predicted_label'] = df['predicted_label'].fillna('None')

    # 5-2. Label Clustering Outputs
    df = label_topic.extract_topic_cosine(df, embeddings, model_selected, exp_name)

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
    nmi_spectral, silhouette_spectral, purity, entropy_score = fc.clustering_performance(df, embeddings, test_data_labeled)

    file_name = "clustering/result/"+ str_embedding_method + " using " + str_reflection + "_" + str(exp_name) + ".txt"
    with open(file_name, "a") as file:
        file.write(f"<<<{str_clustering_method}>>>")
        file.write(f"\n# Weighted Averages of Error and Correctness of {str_clustering_method} with {str_embedding_method}\n")
        file.write(df_correctness_clst.to_string(index=False, header=True))

        file.write(f"\n# Evaluation of {str_clustering_method} with {str_embedding_method}")
        # file.write(f"\nNMI: {nmi_spectral:.2f}")
        file.write(f"\n- Silhouette Score: {silhouette_spectral:.2f}")
        file.write(f'\n- Purity: {purity:.2f}')
        file.write(f'\n- Entropy Score: {entropy_score:.2f}')
        file.write("\n------------------------------\n")


from sklearn.cluster import KMeans
def elbow_method(embeddings):
    sse = []
    k_range = range(1, 16)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        sse.append(kmeans.inertia_)

    sse_diff = np.diff(sse)
    sse_diff_diff = np.diff(sse_diff)
    optimal_k = np.argmin(sse_diff_diff) + 2
    count_label = optimal_k

    return count_label


def execute_clustering(embeddings, str_embedding_method, str_reflection, processed_df_cluster, count_label):
    str_clustering_method = "Spectral Clustering"
    clustered_data_sp = fc.clustering_spectral(embeddings, processed_df_cluster, count_label)
    clustering_evaluation(embeddings, clustered_data_sp, str_clustering_method, str_embedding_method, str_reflection, count_label, "w")

    str_clustering_method = "Affinity Propagation Clustering"
    clustered_data_ap = fc.clustering_affinity_propagation(embeddings, processed_df_cluster, count_label)
    clustering_evaluation(embeddings, clustered_data_ap, str_clustering_method, str_embedding_method, str_reflection, count_label)

    str_clustering_method = "K-means Clustering"
    clustered_data_km = fc.clustering_kmeans(embeddings, processed_df_cluster, count_label)
    clustering_evaluation(embeddings, clustered_data_km, str_clustering_method, str_embedding_method, str_reflection, count_label)

    print("\n---------------------------------------------------")
    print(f"***** {str_embedding_method} for {str_reflection} done")
    print("---------------------------------------------------\n")


def execute_all(cluster_data, str_reflection, model_selected):

    # for embedding_method in [1, 2, 3, 4]:
        embedding_method = 4
        processed_df_cluster, count_label, class_weights_tensor, unique_labels = fc.load_data(data_path, cluster_data, label_exist=False, test_data_labeled=True, train=True)
        processed_df_cluster['original_labels'] = processed_df_cluster['original_labels'].fillna('None')
        embeddings, str_embedding_method = fc.generate_embeddings(embedding_method, processed_df_cluster, count_label, model_selected, exp_name)
        # count_label = 10
        # count_label = elbow_method(embeddings)
        # print(f"The number of Labels by Elbow method: {count_label}")
        execute_clustering(embeddings, str_embedding_method, str_reflection, processed_df_cluster, count_label)


if(int_ref_num == 0):
    cluster_data = sheet_names
    str_reflection = "Entire Reflection"
    execute_all(cluster_data, str_reflection, model_selected)
else:
    for i in range(len(sheet_names)):
        cluster_data = [sheet_names[i]]
        str_reflection = "Reflection " + str(i + 1)
        execute_all(cluster_data, str_reflection, model_selected)
