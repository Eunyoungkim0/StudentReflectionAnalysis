from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch

import data_preprocessing as dpp
import functions as fc

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import os
import shutil
from datasets import Dataset

main_dir = "."
# data = os.path.join(main_dir, "Data", "Consensus_Data_One_Label.xlsx")
data = os.path.join(main_dir, "Data", "Consensus_Data_One_Label_EK11-24.xlsx")

result_dir = os.path.join(main_dir, "Results")
finetune_dir = os.path.join(main_dir, "finetuned_distilbert")

if os.path.exists(result_dir):
    shutil.rmtree(result_dir)

if os.path.exists(finetune_dir):
    shutil.rmtree(finetune_dir)

sheet_names = ['D-ESP4-1','D-ESU4-1','D-ESP4-2','D-ESU4-2','D-ESP4-3','D-ESU4-3','D-ESP4-4','D-ESU4-4']
train_sheet_names = ['D-ESP4-1','D-ESP4-2','D-ESP4-3','D-ESP4-4']
test_sheet_names  = ['D-ESU4-1','D-ESU4-2','D-ESU4-3','D-ESU4-4']

class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.class_weights = class_weights
        self.class_weights = class_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        # labels = inputs.get("labels")
        labels = inputs.get("labels").to(self.model.device)
        # outputs = model(**inputs)
        outputs = model(**{k: v.to(self.model.device) for k, v in inputs.items()})
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


start_time = datetime.now()
print(f"start time: {start_time}")

# 1. Load Data
df_ref = dpp.make_dataset(data, train_sheet_names) # Train data
df_ref_test = dpp.make_dataset(data, test_sheet_names) # Test data

# 2. Data pre-processing
column_input = "text"
column_answer = "labels"

# Train data
processed_df, count_label, class_weights_tensor = dpp.data_preprocessing(df_ref, column_input, column_answer)
dataset = Dataset.from_pandas(processed_df)

# Test data
processed_df_test, count_label_test, class_weights_tensor_test = dpp.data_preprocessing(df_ref_test, column_input, column_answer)
dataset_test = Dataset.from_pandas(processed_df_test)

# 3. Tokenization
tokenized_dataset = dataset.map(fc.tokenizer_function, batched=True, fn_kwargs={'finetuned': False})
tokenized_dataset_test = dataset_test.map(fc.tokenizer_function, batched=True, fn_kwargs={'finetuned': False})

# 4. Create model
model = fc.create_model(count_label)

training_args = TrainingArguments(
    output_dir=result_dir,
    learning_rate=8e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=7,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True, # default: lowest validation loss
    dataloader_drop_last=False
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset_test,
    tokenizer=fc.tokenizer,
    data_collator=fc.data_collator,
    compute_metrics=fc.compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
    class_weights=class_weights_tensor
)

# 5. Train the model
trainer.train()

# 6. Confusion matrix
predictions = trainer.predict(tokenized_dataset_test)
pred_labels = np.argmax(predictions.predictions, axis=-1)

conf_matrix = confusion_matrix(predictions.label_ids, pred_labels)

plt.figure(figsize=(6, 4))
unique_labels = processed_df[['original_labels', 'labels']].drop_duplicates()
sorted_labels = unique_labels.sort_values(by='labels')
original_labels = sorted_labels['original_labels'].tolist()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=original_labels, yticklabels=original_labels)

plt.xticks(fontsize=8, rotation=45, ha='right')
plt.yticks(fontsize=8)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap - BERT")
plt.show()

# 7. Correctness
df_correctness = fc.correctness(conf_matrix, original_labels)
print(df_correctness)

# 8. Save the model
fc.save_finetuned(model, fc.tokenizer)

end_time = datetime.now()
elapsed_time = end_time - start_time

print("\n-------------------------")
print(f"Start time: {start_time}")
print(f"End   time: {end_time}")
print(f"Execution Duration:     {elapsed_time}")