import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datasets import Dataset

import data_preprocessing as dpp
import functions as fc

from sklearn.metrics import confusion_matrix
from sklearn.cluster import SpectralClustering

main_dir = "."
data = os.path.join(main_dir, "Data", "Consensus_Data_One_Label_EK11-24.xlsx")

sheet_names = ['D-ESP4-1','D-ESU4-1','D-ESP4-2','D-ESU4-2','D-ESP4-3','D-ESU4-3','D-ESP4-4','D-ESU4-4']
train_sheet_names = ['D-ESP4-1','D-ESP4-2','D-ESP4-3','D-ESP4-4']
test_sheet_names  = ['D-ESU4-1','D-ESU4-2','D-ESU4-3','D-ESU4-4']

# 1. Load data
df_ref_cluster = dpp.make_dataset(data, test_sheet_names)

column_input = "text"
column_answer = "labels"

processed_df_cluster, count_label, class_weights_tensor = dpp.data_preprocessing(df_ref_cluster, column_input, column_answer)
dataset_cluster = Dataset.from_pandas(processed_df_cluster)

tokenized_dataset_cluster = dataset_cluster.map(fc.tokenizer_function, batched=True, fn_kwargs={'finetuned': True})

# 2. Generate embedding from the fine-tuned model
embedded_dataset = tokenized_dataset_cluster.map(fc.generate_embeddings, batched=True, batch_size=16, fn_kwargs={'finetuned': False, 'count_label': count_label})

embeddings_array = np.array(embedded_dataset['embeddings'], dtype=np.float32)
embeddings_cls = embeddings_array[:, 0, :]

# 3. Clustering
spectral = SpectralClustering(n_clusters=count_label+1, affinity='cosine') # Using cosine similarity
labels = spectral.fit_predict(embeddings_cls)
processed_df_cluster['cluster'] = labels

# 4. Output
pd.set_option('display.max_colwidth', 90)
pd.set_option('display.width', 150)

for cluster, group in processed_df_cluster.groupby('cluster'):
   print(f"Cluster {cluster}:")
   print(group[['text', 'original_labels', 'cluster']])
   print("\n")


# 5. Confusion matrix
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
cluster_to_label = (
    processed_df_cluster.groupby('cluster')['original_labels']
    .agg(lambda x: x.value_counts().idxmax())
    .to_dict()
)

processed_df_cluster['predicted_label'] = processed_df_cluster['cluster'].map(cluster_to_label)
true_labels_encoded = label_encoder.fit_transform(processed_df_cluster['original_labels'])
pred_labels_encoded = label_encoder.transform(processed_df_cluster['predicted_label'])

conf_matrix_clst = confusion_matrix(true_labels_encoded, pred_labels_encoded)

conf_matrix_df = pd.DataFrame(
    conf_matrix_clst,
    index=label_encoder.classes_,
    columns=label_encoder.classes_
)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", cbar=True)

plt.xticks(fontsize=8, rotation=45, ha='right')
plt.yticks(fontsize=8)
plt.title("Confusion Matrix Heatmap - Clustering")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

plt.show()

# 6. Correctness
unique_labels = processed_df_cluster[['original_labels', 'labels']].drop_duplicates()
sorted_labels = unique_labels.sort_values(by='labels')
original_labels = sorted_labels['original_labels'].tolist()
df_correctness_clst = fc.correctness(conf_matrix_clst, original_labels)
print(df_correctness_clst)


# 7. Clustering performance evaluation
nmi_spectral, silhouette_spectral, purity = fc.clustering_performance(processed_df_cluster)
print(f"NMI (Spectral): {nmi_spectral}")
print(f"Silhouette Score (Spectral): {silhouette_spectral}")
print(f'Purity: {purity:.2f}')


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