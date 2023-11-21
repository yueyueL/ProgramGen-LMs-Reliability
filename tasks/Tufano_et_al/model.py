import argparse
import os
import logging
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Dataset
import random
import numpy as np
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer, T5EncoderModel,T5Config,
                          AutoTokenizer, AutoModelForSeq2SeqLM)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer), 'codet5': (T5Config, T5EncoderModel, RobertaTokenizer)}

cpu_cont = 16
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source_ids,
                 target_ids,
                 ):
        self.idx = idx
        self.source_ids = source_ids
        self.target_ids = target_ids

def convert_examples_to_features(idx, source, target, tokenizer, args):
    # encode - subword tokenize
    source_ids = tokenizer.encode(source, truncation=True, max_length=args.encoder_block_size, padding='max_length', return_tensors='pt')
    target_ids = tokenizer.encode(target, truncation=True, max_length=args.decoder_block_size, padding='max_length', return_tensors='pt')
    # label = tokenizer.encode(label, truncation=True, max_length=args.decoder_block_size, padding='max_length', return_tensors='pt')
    return InputFeatures(idx, source_ids, target_ids)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type):
        if file_type == 'train':
            file_path = args.train_data_file
        elif file_type == 'eval':
            file_path = args.eval_data_file
        elif file_type == 'test':
            file_path = args.test_data_file
        
        self.examples = []
        source_path = file_path + ".code_before.txt"
        target_path = file_path + ".code_after.txt"

        with open(source_path, "rb") as f:
            sources = f.readlines()
        with open(target_path, "rb") as f:
            targets = f.readlines()
        
        sources = [s.decode('utf-8').strip() for s in sources]
        targets = [t.decode('utf-8').strip() for t in targets]

        for i in range(len(sources)):
            source = sources[i]
            target = targets[i]
            if len(source) == 0 or len(target) == 0:
                continue
            self.examples.append(convert_examples_to_features(i, source, target, tokenizer, args))

        # if file_type == 'train':
        #     for example in self.examples:
        #         logger.info("*** Example ***")
        #         logger.info("index: {}".format(example.idx))
        #         logger.info("input_ids: {}".format(' '.join(map(str, example.source_ids))))
        #         logger.info("decoder_input_ids: {}".format(' '.join(map(str, example.decoder_input_ids)))) 
                
    def __getitem__(self, i):
        return self.examples[i].source_ids, self.examples[i].source_ids.ne(0), self.examples[i].target_ids, self.examples[i].target_ids.ne(0)

    def __len__(self):
        return len(self.examples)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens

def train(args, train_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)
    
    args.max_steps = args.epochs * len(train_dataloader)

    # evaluate model per epoch
    args.save_steps = len(train_dataloader) * 1
   
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    
    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_loss = 100

    model.zero_grad()

    for idx in range(args.epochs): 
        #bar = tqdm(train_dataloader, total=len(train_dataloader))
        logger.info(" Epoch: %d", idx)
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(train_dataloader):
            (input_ids, attention_mask, labels, decoder_attention_mask) = [x.squeeze(1).to(args.device) for x in batch]
        
            model.train()
            # the forward function automatically creates the correct decoder_input_ids
            outputs_all = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, decoder_attention_mask = decoder_attention_mask, output_attentions = True)
            loss = outputs_all.loss
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num,5)
            #bar.set_description("epoch {} loss {}".format(idx,avg_loss))
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if global_step % args.save_steps == 0:
                    # placeholder of evaluation
                    eval_loss = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)    
                    # Save model checkpoint
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        logger.info("  "+"*"*20)  
                        logger.info("  Best Loss:%s",round(best_loss,4))
                        logger.info("  "+"*"*20)                          
                        checkpoint_prefix = 'checkpoint-best-loss'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name)) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    
    eval_loss, num = 0, 0
    #bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch in eval_dataloader:
        (input_ids, attention_mask, labels, decoder_attention_mask) = [x.squeeze(1).to(args.device) for x in batch]
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, decoder_attention_mask=decoder_attention_mask ).loss
        if args.n_gpu > 1:
            loss = loss.mean()
        eval_loss += loss.item()
        num += 1
    eval_loss = round(eval_loss/num,5)
    model.train()
    logger.info("***** Eval results *****")
    logger.info(f"Evaluation Loss: {str(eval_loss)}")
    return eval_loss

def test(args, model, tokenizer, test_dataset):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Test!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    model.eval()
    accuracy = []
    raw_predictions = []
    correct_prediction = ""
    #bar = tqdm(test_dataloader, total=len(test_dataloader))
    for batch in test_dataloader:
        correct_pred = False
        (input_ids, attention_mask, decoder_input_ids, _)=[x.squeeze(1).to(args.device) for x in batch]
        with torch.no_grad():
            beam_outputs = model.generate(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          do_sample=False, # disable sampling to test if batching affects output
                                          num_beams=args.num_beams,
                                          num_return_sequences=args.num_beams,
                                          max_length=args.decoder_block_size)
        beam_outputs = beam_outputs.detach().cpu().tolist()
        decoder_input_ids = decoder_input_ids.detach().cpu().tolist()
        for single_output in beam_outputs:
            # pred
            prediction = tokenizer.decode(single_output, skip_special_tokens=False)
            prediction = clean_tokens(prediction)
            # truth
            ground_truth = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=False)
            ground_truth = clean_tokens(ground_truth)

            if prediction == ground_truth:
                correct_prediction = prediction
                correct_pred = True
                break
        if correct_pred:
            raw_predictions.append(correct_prediction)
            accuracy.append(1)
        else:
            # if not correct, use the first output in the beam as the raw prediction
            raw_pred = tokenizer.decode(beam_outputs[0], skip_special_tokens=False)
            raw_pred = clean_tokens(raw_pred)
            raw_predictions.append(raw_pred)
            accuracy.append(0)
        nb_eval_steps += 1
        
    # calculate accuracy
    test_result = round(sum(accuracy) / len(accuracy), 4)
    logger.info("***** Test results *****")
    logger.info(f"Test Accuracy: {str(test_result)}")

    # write prediction to file
    with open(os.path.join(args.output_dir, "predictions.txt"), "w+") as writer:
        for prediction in raw_predictions:
            writer.write(prediction + "\n")

    with open(os.path.join(args.output_dir, "test_result.txt"), "w+") as writer:
        for result in accuracy:
            writer.write(str(result) + "\n")


def main():
    parser = argparse.ArgumentParser()
    # Params
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="t5", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--encoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--decoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--num_beams", default=1, type=int,
                        help="Beam size to use when decoding.")                          
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--checkpoint_model_name", default="non_domain_model.bin", type=str,
                            help="Checkpoint model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_explain", action='store_true',
                        help="Whether to run explain on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    args = parser.parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    args.device = device
    print(device)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)
    # Set seed
    set_seed(args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model=AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, file_type='eval')
        train(args, train_dataset, model, tokenizer, eval_dataset)
    # Evaluation
    results = {}  
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-loss/'+args.model_name
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        test(args, model, tokenizer, test_dataset)

    return results

if __name__ == "__main__":
    main()


