
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "1" # use the gpu number 1
device ="cuda"

# import packages
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer,AutoTokenizer
from transformers import DataCollatorWithPadding
from datasets import load_dataset
import json
from sklearn.metrics import classification_report



# Download dataset
imdb = load_dataset("./hateval") # here the path to the dataset
model_type = "GroNLP/hateBERT"  # here is the hugging-face model
tokenizer = AutoTokenizer.from_pretrained(model_type,max_length=100)


# Preprocess dataset
def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True)

tokenized_imdb = imdb.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


# Function to compute the eval metric of the model
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    res = classification_report(labels, predictions, digits=4, output_dict=True)
    with open('./BERTHatEval.json', 'w') as fp:  # the output evaluation metric to be written in ./BERTHatEval.json
        json.dump(res, fp)
    macroF1 = res['macro avg']["f1-score"]
    F1posclass = res["1"]["f1-score"]
    finalRes = dict()
    finalRes["macro-average-F1"] = macroF1
    finalRes["F1-positive-class"] = F1posclass
    return finalRes


# Define the model
model = AutoModelForSequenceClassification.from_pretrained(
    model_type, num_labels=2, id2label=id2label, label2id=label2id
)


# Define the training args
training_args = TrainingArguments(
    output_dir="./hatebert_model_fine_tuned_on_hateval/", # specify the output_dir here
    learning_rate=1e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0,
    lr_scheduler_type='constant',
    save_strategy="no",
   )


# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# Train and save the model
trainer.train()
trainer.save_model('./teacher/hateval') # specify the path where to save the final trained model


# Eval
model_checkpoint = './teacher/hateval' # specify the model to load for evaluation
model_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model_finetuned = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,num_labels=2, id2label=id2label, label2id=label2id)

OUTPUT_DIR = "./output" # specify the second trainer output_dir
test_args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    do_train = False,
    do_eval = True,
    per_device_eval_batch_size =32,
    dataloader_drop_last = False)

trainer2 = Trainer(
    model=model_finetuned,
    args=test_args,
    tokenizer = model_tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics)

print(trainer2.evaluate(tokenized_imdb["test"]))

