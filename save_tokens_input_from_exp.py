import json
from ecco import from_pretrained
import torch
import os
import numpy as np
import pickle

def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    tokens = tokens.replace(" ", "")
    return tokens

def comparing_all_exsiting_tokens(all_tokens,  exps_lists, top_remove=5, reverse=False, is_random=False):

    reshapred_exps = [sublist[:len(exps_lists[0])] for sublist in exps_lists]
    average_exps = np.average(reshapred_exps, axis=0)

    if reverse:
        sorted_index = np.argsort(-average_exps)
    else:
        sorted_index = np.argsort(average_exps)

    if is_random:
        sorted_index = np.random.permutation(len(average_exps))

    all_input_tokens = all_tokens[:len(exps_lists[0])]
    # replace the tokens small with " "
    input_tokens_without_small = []
    removed_index = []
    start_index = 0
    while len(removed_index) < top_remove:
        if start_index >= len(sorted_index):
            break
        if sorted_index[start_index] not in removed_index and all_input_tokens[sorted_index[start_index]] not in ["<pad>", "<s>", "</s>"]:
            removed_index.append(sorted_index[start_index])
        start_index += 1

    for i in range(len(all_input_tokens)):
        if i in removed_index:
            input_tokens_without_small.append(" "*len(all_input_tokens[i]))
        else:
            input_tokens_without_small.append(all_input_tokens[i])

    return input_tokens_without_small


def runing_explaining_by_grad_shap(model_name_or_path, model_path, test_source_lines, test_target_lines, exp_saved_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_config = {
        'embedding': 'shared.weight',
        'type': 'enc-dec',
        'activations': ['wo'],
        'token_prefix': '',
        'partial_token_prefix': '##'
    }

    if model_name_or_path in ["Salesforce/codet5-base", "microsoft/codereviewer", "razent/cotext-1-ccg", "Salesforce/codet5p-220m"]:
        lm = from_pretrained(model_name_or_path, local_model_path=model_path, model_config = model_config, attention=True, device = device)
    else:
        lm = from_pretrained(model_name_or_path, local_model_path=model_path, attention=True, device = device)

    attribution_method = 'grad_shap'
    small_saved_dir = exp_saved_dir
    exp_saved_dir = exp_saved_dir + attribution_method + "/"

    correct_counts = []
    correct_after_remove_small = 0
    new_sources_random, new_sources_bad_1, new_sources_bad_3, new_sources_bad_5, new_sources_bad_10, new_sources_top_5,new_targets = [], [], [], [], [], [], []
    for i in range(len(test_source_lines)):
        exp_path = exp_saved_dir+ "/" + str(i) + ".pkl"
            
        if not os.path.exists(exp_path):
            continue
        exp = read_exps(exp_path)
        if exp is None:
            continue
        values = exp["attribution"]
        all_tokens = exp["tokens"][0]
        source = test_source_lines[i].strip()
        target = test_target_lines[i].strip()
        torch.cuda.empty_cache()

        raw_tokens = lm.tokenizer.tokenize(source)
        target_tokens = lm.tokenizer.tokenize(target)
        target_tokens.insert(0, "<s>")
        target_tokens.append("</s>")
        predicted_tokens = all_tokens[len(values[0]):]

        input_tokens_without_random = comparing_all_exsiting_tokens(all_tokens, values, top_remove=5, reverse=False, is_random=True)
        input_tokens_without_bad_1 = comparing_all_exsiting_tokens(all_tokens, values, top_remove=1, reverse=False, is_random=False)
        input_tokens_without_bad_3 = comparing_all_exsiting_tokens(all_tokens, values, top_remove=3, reverse=False, is_random=False)
        input_tokens_without_bad_5 = comparing_all_exsiting_tokens(all_tokens, values, top_remove=5, reverse=False, is_random=False)
        input_tokens_without_bad_10 = comparing_all_exsiting_tokens(all_tokens, values, top_remove=10, reverse=False, is_random=False)
        input_tokens_without_top_5 = comparing_all_exsiting_tokens(all_tokens, values, top_remove=5, reverse=True, is_random=False)

        new_input_string_random = lm.tokenizer.convert_tokens_to_string(input_tokens_without_random).replace("</s><pad>", "")
        new_input_string_random = new_input_string_random.replace("<s>", "")

        new_input_string_bad_1 = lm.tokenizer.convert_tokens_to_string(input_tokens_without_bad_1).replace("</s><pad>", "")
        new_input_string_bad_1 = new_input_string_bad_1.replace("<s>", "")

        new_input_string_bad_5 = lm.tokenizer.convert_tokens_to_string(input_tokens_without_bad_5).replace("</s><pad>", "")
        new_input_string_bad_5 = new_input_string_bad_5.replace("<s>", "")

        new_input_string_bad_3 = lm.tokenizer.convert_tokens_to_string(input_tokens_without_bad_3).replace("</s><pad>", "")
        new_input_string_bad_3 = new_input_string_bad_3.replace("<s>", "")

        new_input_string_bad_10 = lm.tokenizer.convert_tokens_to_string(input_tokens_without_bad_10).replace("</s><pad>", "")
        new_input_string_bad_10 = new_input_string_bad_10.replace("<s>", "")

        new_input_string_top_5 = lm.tokenizer.convert_tokens_to_string(input_tokens_without_top_5).replace("</s><pad>", "")
        new_input_string_top_5 = new_input_string_top_5.replace("<s>", "")

        new_sources_random.append(new_input_string_random)
        new_sources_bad_1.append(new_input_string_bad_1)
        new_sources_bad_5.append(new_input_string_bad_5)
        new_sources_bad_3.append(new_input_string_bad_3)
        new_sources_bad_10.append(new_input_string_bad_10)
        new_sources_top_5.append(new_input_string_top_5)
        new_targets.append(target)

        predicted_string = lm.tokenizer.convert_tokens_to_string(predicted_tokens)

        if clean_tokens(target, model_type="codet5") == clean_tokens(predicted_string, model_type="codet5"):
            correct_counts.append(1)
        else:
            correct_counts.append(0)



    # #save random inputs
    smaller_inputs_dir = small_saved_dir + "smaller_inputs_random/"
    if not os.path.exists(smaller_inputs_dir):
        os.makedirs(smaller_inputs_dir)        
    with open(smaller_inputs_dir + "test.code_before.txt", "w") as f:
        f.write("\n".join(new_sources_random))
    with open(smaller_inputs_dir + "test.code_after.txt", "w") as f:
        f.write("\n".join(new_targets))

    # #save bad_1 inputs
    smaller_inputs_dir = small_saved_dir + "smaller_inputs_bad_1/"
    if not os.path.exists(smaller_inputs_dir):
        os.makedirs(smaller_inputs_dir)
    with open(smaller_inputs_dir + "test.code_before.txt", "w") as f:
        f.write("\n".join(new_sources_bad_1))
    with open(smaller_inputs_dir + "test.code_after.txt", "w") as f:
        f.write("\n".join(new_targets))
    
    # #save bad_5 inputs
    smaller_inputs_dir = small_saved_dir + "smaller_inputs_bad_5/"
    if not os.path.exists(smaller_inputs_dir):
        os.makedirs(smaller_inputs_dir)
    with open(smaller_inputs_dir + "test.code_before.txt", "w") as f:
        f.write("\n".join(new_sources_bad_5))
    with open(smaller_inputs_dir + "test.code_after.txt", "w") as f:
        f.write("\n".join(new_targets))

    # #save bad_3 inputs
    smaller_inputs_dir = small_saved_dir + "smaller_inputs_bad_3/"
    if not os.path.exists(smaller_inputs_dir):
        os.makedirs(smaller_inputs_dir)
    with open(smaller_inputs_dir + "test.code_before.txt", "w") as f:
        f.write("\n".join(new_sources_bad_3))
    with open(smaller_inputs_dir + "test.code_after.txt", "w") as f:
        f.write("\n".join(new_targets))

    # #save bad_10 inputs
    smaller_inputs_dir = small_saved_dir + "smaller_inputs_bad_10/"
    if not os.path.exists(smaller_inputs_dir):
        os.makedirs(smaller_inputs_dir)
    with open(smaller_inputs_dir + "test.code_before.txt", "w") as f:
        f.write("\n".join(new_sources_bad_10))
    with open(smaller_inputs_dir + "test.code_after.txt", "w") as f:
        f.write("\n".join(new_targets))
    
    # #save top_5 inputs
    smaller_inputs_dir = small_saved_dir + "smaller_inputs_top_5/"
    if not os.path.exists(smaller_inputs_dir):
        os.makedirs(smaller_inputs_dir)
    with open(smaller_inputs_dir + "test.code_before.txt", "w") as f:
        f.write("\n".join(new_sources_top_5))
    with open(smaller_inputs_dir + "test.code_after.txt", "w") as f:
        f.write("\n".join(new_targets))


        
def read_exps(exp_p):
    with open(exp_p, "rb") as f:
        try :
            exp = pickle.load(f)
            if isinstance(exp, dict):
                return exp
            else:
                print(exp)
        except:
            print(f"Problematic file: {exp_p}")
            print(f"File size: {os.path.getsize(exp_p)} bytes")
            os.remove(exp_p)
    return None


def running_comparing_source_targets():
    model_names = ["microsoft/codereviewer", "Salesforce/codet5p-220m", "Salesforce/codet5-base"]
    saved_model_names = ["codereviewer", "codet5p-220m", "codet5"]


    ## code review, tufano_et_al
    data_types = ["android", "google","ovirt"]
    data_sizes = ["small", "medium"]
    output_dir = "/path/to/tufano_et_al/output/"
    data_dir = "/path/to/tufano_et_al/data/"
    for data_type in data_types:
        for data_size in data_sizes:
            for model_name, saved_model_name in zip(model_names, saved_model_names):
                checkpoint_prefix = 'checkpoint-best-loss/model.bin'
                model_path = output_dir + data_type + "/" + data_size + "/" + saved_model_name + "/" + checkpoint_prefix
                exp_saved_dir = output_dir + data_type + "/" + data_size + "/" + saved_model_name + "/new_exps/"

                test_data_file = data_dir + data_type + "/" + data_size + "/"
                test_file_source_p = test_data_file + "test.code_before.txt" 
                test_file_target_p = test_data_file + "test.code_after.txt"
                
                with open(test_file_source_p, 'r') as f:
                    test_source_lines = f.readlines()
                with open(test_file_target_p, 'r') as f:
                    test_target_lines = f.readlines()   
                runing_explaining_by_grad_shap(model_name, model_path, test_source_lines, test_target_lines, exp_saved_dir)     

if __name__ == '__main__':
    running_comparing_source_targets()





