

import os
os.environ["CUDA_VISIBLE_DEVICES"]= "5"
device ="cuda"
from sklearn.metrics import classification_report
import torch
import numpy as np
import datasets
import transformers
datasets.logging.set_verbosity_error()
transformers.logging.set_verbosity_error()
from datasets import load_dataset
import torch.nn.functional as F
import argparse
import statistics
import json
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                         TrainingArguments, Trainer, AutoConfig, DataCollatorWithPadding)



# Initialize the data
def get_student(args):
    s_config = AutoConfig.from_pretrained(args.student_model_name_or_path)
    s_config.num_hidden_layers = args.num_hidden_layers
    s_model = AutoModelForSequenceClassification.from_pretrained(args.student_model_name_or_path, config=s_config)
    return s_model


# Functions to process the data
student_tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")
def preprocess_function(examples):
    return student_tokenizer(examples["sentence"], padding="max_length", max_length=128,truncation=True)

def convert_examples_to_features(hateval,num_train_examples, num_eval_examples):
    train_dataset = (hateval['train']
                    .select(range(num_train_examples))
                    .map(preprocess_function, batched=True))
    eval_dataset = (hateval['test']
                    .select(range(num_eval_examples))
                    .map(preprocess_function, batched=True))
    train_labels = torch.tensor(hateval["train"]["label"][:num_train_examples])
    test_labels =  torch.tensor(hateval["test"]["label"][:num_eval_examples])
    eval_examples = hateval['test'].select(range(num_eval_examples))
    return train_dataset, eval_dataset, eval_examples, train_labels, test_labels



# Function to compute the evaluation metric
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    res = classification_report(labels, predictions, digits=4, output_dict=True)
    # with open('./res.json', 'w') as fp:
    #     json.dump(res, fp)
    macroF1 = res['macro avg']["f1-score"]
    F1posclass = res["1"]["f1-score"]
    finalRes = dict()
    finalRes["macro-average-F1"] = macroF1
    finalRes["F1-positive-class"] = F1posclass
    return finalRes


# Introduce the distillation training arguments
class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha_ce=0.5, alpha_soft=0.5, alpha_mse=0.01, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_ce = alpha_ce  # contribution of the logit loss
        self.alpha_soft = alpha_soft  # contribution of the hard loss
        self.alpha_mse = alpha_mse  # contribution of the mse loss
        self.temperature = temperature


# Distillation Trainer
class DistillationTrainer(Trainer):
    def __init__(self, *args, k, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.KNN=k
        self.teacher.eval()  # teacher is in the eval mode
        self.train_dataset.set_format(
            type=self.train_dataset.format["type"], columns=list(self.train_dataset.features.keys()))

    # Function to compute the distillation loss function
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs_stu = {
            "input_ids": inputs['input_ids'],
            "attention_mask": inputs['attention_mask'],

        }
        outputs_stu = model(**inputs_stu, labels=inputs["labels"].unsqueeze(0),
                            output_hidden_states=True)  # model takes the input and provide output
        loss = outputs_stu.loss
        with torch.no_grad():
            outputs_tea = self.teacher(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"], output_hidden_states=True)  # , labels=inputs["labels"]

        t_logits, t_features = outputs_tea[0], outputs_tea[-1]
        train_loss, s_logits, s_features = outputs_stu[0], outputs_stu[1], outputs_stu[-1]
        soft_targets = F.softmax(t_logits / self.args.temperature, dim=-1)
        log_probs = F.log_softmax(s_logits / self.args.temperature, dim=-1)
        soft_loss = F.kl_div(log_probs, soft_targets.detach(),
                             reduction='batchmean') * self.args.temperature * self.args.temperature
        outputs_stu_hidden_states = outputs_stu.hidden_states
        outputs_tea_hidden_states = outputs_tea.hidden_states
        s_hidden_states = outputs_stu_hidden_states[-1]  # (bs, seq_length, dim)
        t_hidden_states = outputs_tea_hidden_states[-1]  # (bs, seq_length, dim)
        s_features = s_hidden_states
        t_features = t_hidden_states
        labels = inputs["labels"]
        dicti = {}
        labels_unique = torch.unique(labels)
        for label in labels_unique:
            dicti[str(label.item())] = (labels == label).nonzero(as_tuple=True)[0]
        distill_loss_curr = 0
        for i in range(s_features.size(0)):

            label = labels[i]
            indices = dicti[str(label.item())]
            elements_orig = torch.index_select(t_features, 0, indices)
            s_feature = torch.reshape(s_features[i], (1, -1))
            elements = torch.reshape(elements_orig, (elements_orig.size(0), -1))
            dist_sub = torch.sub(s_feature, elements)
            # dist = torch.mean(dist_sub**2,1)
            dist = torch.sum(dist_sub ** 2, 1)
            sorted_dist, indices_to_use = torch.sort(dist)
            if sorted_dist.size(0) < self.KNN:
                self.KNN = sorted_dist.size(0)
            indices_to_use = indices_to_use[:self.KNN]
            final_elements = torch.index_select(elements_orig, 0, indices_to_use)
            curr_Loss = F.mse_loss(final_elements, s_features[i].unsqueeze(0), reduction='none').mean(dim=(1, 2))
            curr_Loss = torch.sum(curr_Loss)
            distill_loss_curr = distill_loss_curr + curr_Loss
        loss_ours = distill_loss_curr
        loss = self.args.alpha_mse * loss_ours + self.args.alpha_soft * soft_loss + self.args.alpha_ce * loss
        return (loss, outputs_stu) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()

    # Required parameters

    parser.add_argument(
        "--student_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )

    parser.add_argument(
        "--teacher_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )

    parser.add_argument(
        "--dataset_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )

    parser.add_argument(
        "--num_hidden_layers", default=6, type=int, help="Total number of hidden layers."
    )

    parser.add_argument(
        "--train_batch_size", default=16, type=int, help="The train bacth size."
    )

    parser.add_argument(
        "--test_batch_size", default=16, type=int, help="The test bacth size."
    )

    parser.add_argument(
        "--sequence_length", default=128, type=int, help="The sequence length."
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--nearest_neighbors", default=2, type=int, help="number of nearest neighbors to consider."
    )
    parser.add_argument(
        "--alpha_ce", default=0.5, type=float, help="Typical cross entropy loss."
    )
    parser.add_argument(
        "--alpha_soft", default=0.5, type=float, help="Soft loss linear weight. Only for distillation."

    )
    parser.add_argument(
        "--alpha_mse", default=0.01, type=float,
        help="Distillation mse embedding loss linear weight. Only for distillation."

    )
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="the student model learning rate."

    )
    parser.add_argument(
        "--temperature", default=2.0, type=float,
        help="temperature for soft diatillation. Only for distillation."

    )

    args = parser.parse_args()
    dataset = load_dataset(args.dataset_path)  # "./hateval"
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model_name_or_path)  # "GroNLP/hateBERT"
    data_collator = DataCollatorWithPadding(tokenizer=student_tokenizer)
    student_model = get_student(args)
    # Process the data
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    num_train_examples = len(dataset['train'])
    num_eval_examples = len(dataset['test'])
    train_ds, eval_ds, eval_examples, train_labels, test_labels = convert_examples_to_features(dataset,
                                                                                               num_train_examples,
                                                                                               num_eval_examples)
    logging_steps = len(train_ds) // args.train_batch_size
    student_training_args = DistillationTrainingArguments(alpha_ce = args.alpha_ce, alpha_soft= args.alpha_soft, alpha_mse=args.alpha_mse, temperature=args.temperature,
        output_dir = "./distiled_models",
        save_strategy="no",
        learning_rate=args.learning_rate,
        lr_scheduler_type='constant',
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        num_train_epochs=3,
        weight_decay=0,
        overwrite_output_dir=True,
        logging_steps=logging_steps,
        do_eval=False,
    )
    print(f"Number of training examples: {train_ds.num_rows}")
    print(f"Number of validation examples: {eval_ds.num_rows}")
    print(f"Number of raw validation examples: {eval_examples.num_rows}")
    print(f"Logging steps: {logging_steps}")

    # Set the teacher model
    teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_model_name_or_path).to(device)

    distil_trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        k= args.nearest_neighbors,
        args=student_training_args,
        train_dataset=train_ds,
        tokenizer=student_tokenizer,
    )

    distil_trainer.train()  # train
    distil_trainer.save_model(args.output_dir)  # save the mode to the specified path "./student/hateval"

    # Eval
    student_model = args.output_dir  # use the save student model
    student_finetuned = AutoModelForSequenceClassification.from_pretrained(student_model, num_labels=2,
                                                                           id2label=id2label, label2id=label2id)
    test_args = TrainingArguments(
        output_dir = "./distiled_models",
        do_train=False,
        do_eval=True,
        save_strategy = "no",
        overwrite_output_dir = True,
        per_device_eval_batch_size=args.test_batch_size,
        dataloader_drop_last=False)
    student_trainer = Trainer(
        model=student_finetuned,
        args=test_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=student_tokenizer)

    # Print the results
    res = student_trainer.evaluate(eval_ds)
    with open("./reshatebert.json", 'w') as fp: # the file path that contains the output results
          json.dump(res, fp)



if __name__ == "__main__":
    main()







