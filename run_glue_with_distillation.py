
# Import Packages
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # Select GPU ID
device = "cuda"
import argparse
import copy
import glob
import logging
import time
import numpy as np
import torch
import json
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP
from distillation import PatientDistillation
import warnings




warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}
log_json = []
def train(args, train_dataset, student_model, teacher_model, d_criterion, tokenizer):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    param_optimizer = list(student_model.named_parameters()) + list(d_criterion.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    student_optimizer = AdamW(optimizer_grouped_parameters, lr=args.student_learning_rate, eps=args.adam_epsilon)
    student_scheduler =get_linear_schedule_with_warmup(student_optimizer, num_warmup_steps=args.warmup_steps,  num_training_steps=t_total)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    student_model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    best_score = 0
    best_model = {
        'epoch': 0,
        'model_state': student_model.state_dict(),
        'optimizer_state': student_optimizer.state_dict()
    }
    for epoch in train_iterator:

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])

        t_start = time.time()

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            student_model.train()
            teacher_model.eval()
            input_ids, attention_mask, token_type_ids, labels = batch[0], batch[1], batch[2], batch[3]
            train_loss, soft_loss, distill_loss = d_criterion(t_model=teacher_model,
                                                              s_model=student_model,
                                                              input_ids=input_ids,
                                                              token_type_ids=token_type_ids,
                                                              attention_mask=attention_mask,
                                                              labels=labels,
                                                              k=args.nearest_neighbors,
                                                              args=args)

            loss = args.alpha * train_loss + (1 - args.alpha) * soft_loss + args.beta * distill_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            student_optimizer.step()
            student_scheduler.step()
            student_optimizer.zero_grad()
            global_step += 1

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        logger.info("***** Epoch: {} *****".format(epoch + 1))
        logger.info("  Train loss: {}".format(tr_loss / len(train_dataset)))
        t_end = time.time()
        logger.info('  Train Time Cost: %.3f' % (t_end - t_start))
        # evaluation
        results = evaluate(args, student_model, tokenizer, prefix='')
        if args.task_name == 'cola':
            eval_score = results['mcc']
        elif args.task_name == 'sst-2':
            eval_score = results['acc']
        elif args.task_name == 'mrpc':
            eval_score = results['acc_and_f1']
        elif args.task_name == 'sts-b':
            eval_score = results['corr']
        elif args.task_name == 'qqp':
            eval_score = results['acc_and_f1']
        elif args.task_name == 'mnli':
            eval_score = results['mnli/acc']
        elif args.task_name == 'mnli-mm':
            eval_score = results['mnli-mm/acc']
        elif args.task_name == 'qnli':
            eval_score = results['acc']
        elif args.task_name == 'rte':
            eval_score = results['acc']
        elif args.task_name == 'wnli':
            eval_score = results['acc']
        else:
            raise NotImplementedError()
        if eval_score > best_score:
            best_score = eval_score
            best_model['epoch'] = epoch + 1
            best_model['model'] = copy.deepcopy(student_model)
        # save checkpoints
        if (args.local_rank in [-1, 0]) and (args.save_epoch > 0 and epoch % args.save_epoch == 0) and (epoch > args.save_after_epoch):
            base_output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch + 1))
            student_output_dir = os.path.join(base_output_dir, 'student')
            for output_dir in [student_output_dir]:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
            student_model_to_save = student_model.module if hasattr(student_model, 'module') else student_model  # Take care of distributed/parallel training
            student_model_to_save.save_pretrained(student_output_dir)
            torch.save(args, os.path.join(student_output_dir, 'training_args.bin'))
            tokenizer.save_pretrained(student_output_dir)
            logger.info("Saving student model checkpoint {0} to {1}".format(epoch + 1, student_output_dir))
        epoch_log = {'epoch': epoch + 1, 'eval_score': eval_score, 'best_score': best_score}
        log_json.append(epoch_log)
        if args.local_rank in [-1, 0]:
            with open(args.output_dir + '/eval_logs.json', 'w') as fp:
                json.dump(log_json, fp)
        t_end = time.time()
        logger.info('Epoch: %d, Train Time: %.3f' % (epoch + 1, t_end - t_start))
        logger.info('********************')
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break
    if args.local_rank in [-1, 0]:  # Save the final model checkpoint
        output_dir = args.output_dir  # ./model/RTE/student
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = best_model['model'].module if hasattr(student_model, 'module') else best_model['model']  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        tokenizer.save_pretrained(output_dir)
        logger.info("Saving the best model checkpoint epoch {} to {}".format(best_model['epoch'], output_dir))
    return global_step, tr_loss / global_step


def compute_BCS(outputs,labels):
    hidden_states = outputs[-1][-1]
    dicti = {}
    labels_unique = torch.unique(labels)
    for label in labels_unique:
        dicti[str(label.item())] = (labels == label).nonzero(as_tuple=True)[0]
    final_cos_sim = 0
    hidden_states_reshaped = hidden_states.view(hidden_states.size(0),-1)
    for i in range(hidden_states_reshaped.size(0)):
        label = labels[i]
        indices = dicti[str(label.item())]
        cos_sim_per_sample = 0
        for j in indices:
            dot_product = torch.sum(hidden_states_reshaped[j,:] *hidden_states_reshaped[i,:], dim=0)
            norm_a = torch.norm(hidden_states_reshaped[j,:], p=2, dim=0)
            norm_b = torch.norm(hidden_states_reshaped[i,:], p=2, dim=0)
            eps = 1e-8
            cos_sim = dot_product / (norm_a * norm_b + eps)
            cos_sim_per_sample = cos_sim_per_sample + cos_sim
        final_cos_sim = final_cos_sim + cos_sim_per_sample/len(indices)
    return final_cos_sim/hidden_states.size(0)


def evaluate(args, model, tokenizer,prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)
    results = {}
    t_start = time.time()
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, eval=True)
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        BSC = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels': batch[3]
                }
                outputs = model(**inputs,output_hidden_states=True)
                BSC.append(compute_BCS(outputs, batch[3]).item())
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        BSC_avg = sum(BSC) / len(BSC)
        eval_loss = eval_loss / nb_eval_steps
        logging.info("  Eval_loss = %f", eval_loss)

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s  , %s  = %s \n" % (key, str(result[key]), "BSC", str(BSC_avg)))
    t_end = time.time()
    logger.info('  Eval Time Cost: %.3f' % (t_end - t_start))

    return results


def load_and_cache_examples(args, task, tokenizer, eval=False):
    if args.local_rank not in [-1, 0] and not eval:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    if eval:
        mode = 'dev'
    else:
        mode = 'train'
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(mode,
                                                                                   list(filter(None, args.student_model.split('/'))).pop(),
                                                                                   str(args.max_seq_length),
                                                                                   str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if eval:
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not eval:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    return dataset


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True, help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--data_dir", default=None, type=str, required=True, help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default=None, type=str, required=True, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    # Other parameters
    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--evaluate_during_training", action='store_true', help="Rul evaluation during training at each logging step.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=0, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=0, help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_epoch', type=int, default=1, help="Save checkpoint every X epochs.")
    parser.add_argument('--save_after_epoch', type=int, default=-1, help="Save checkpoint after epoch.")
    parser.add_argument("--eval_all_checkpoints", action='store_true', help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true', help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument("--teacher_model", default=None, type=str, help="The teacher model dir.")
    parser.add_argument("--student_model", default=None, type=str, required=True, help="The student model dir.")
    parser.add_argument('--alpha', default=0.5, type=float, help="Vanilla knowledge distillation loss radio.")
    parser.add_argument('--beta', default=0.01, type=float, help="feature distillation loss contribution.")
    parser.add_argument("--temperature", default=5.0, type=float, help="Distillation temperature for soft target.")
    parser.add_argument('--num_hidden_layers', default=6, type=int, help="The number of layers of the student model.")
    parser.add_argument('--nearest_neighbors', default=3, type=int, help="The number of nearest neighbors to consider.")
    parser.add_argument("--student_learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam of Student model.")
    parser.add_argument("--strategy", default="first", type=str, help="first | last | skip | both")
    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s", args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % args.task_name)
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    logger.info('Task Name: {}, #Labels: {}'.format(args.task_name, num_labels))

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(
        args.teacher_model,
        do_lower_case=args.do_lower_case,
    )
    teacher_config = config_class.from_pretrained(
        args.teacher_model,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )
    teacher_model = model_class.from_pretrained(
        args.teacher_model,
        from_tf=bool('.ckpt' in args.teacher_model),
        config=teacher_config,
    )
    student_config = config_class.from_pretrained(
        args.student_model,
        num_hidden_layers=args.num_hidden_layers,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )
    student_model = model_class.from_pretrained(
        args.student_model,
        from_tf=bool('.ckpt' in args.student_model),
        config=student_config,
    )

    d_criterion = PatientDistillation(teacher_config, student_config)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    teacher_total_params = sum(p.numel() for p in teacher_model.parameters())
    logger.info('Teacher Model Parameters: {}'.format(teacher_total_params))
    student_total_params = sum(p.numel() for p in student_model.parameters())
    logger.info('Student Model Parameters: {}'.format(student_total_params))
    '''
    04/05/2022 13:13:49 - INFO - __main__ - Teacher Model Parameters: 109483778
    04/05/2022 13:13:49 - INFO - __main__ - Student Model Parameters: 66956546
    '''

    teacher_model.to(args.device)
    student_model.to(args.device)
    d_criterion.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer)
        global_step, tr_loss = train(args, train_dataset, student_model, teacher_model, d_criterion, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]  # ['./model/RTE/student/']
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer,  prefix=global_step)



if __name__ == "__main__":
     main()

