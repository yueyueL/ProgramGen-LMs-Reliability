import json
from ecco import from_pretrained
import torch
import os
import numpy as np
import pickle
import pynvml

def small_attri_array_reshape(attri_array):
    new_attri = []
    for attri in attri_array:
        new_attri.append(attri[:len(attri_array[0])-1])
        
    return np.array(new_attri)


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
    exp_saved_dir = exp_saved_dir + attribution_method + "/"
    if not os.path.exists(exp_saved_dir):
        os.makedirs(exp_saved_dir)

    for i in range(len(test_source_lines)):
        if os.path.exists(exp_saved_dir+ "/" + str(i) + ".pkl"):
            continue
        source = test_source_lines[i].strip()
        target = test_target_lines[i].strip()
        torch.cuda.empty_cache()

        source_ids = lm.tokenizer.encode(source)
        raw_tokens = lm.tokenizer.convert_ids_to_tokens(source_ids)
        saved_json = {}
        if len(raw_tokens) < 500:
            output = lm.generate(source, max_length= 1000, do_sample = False, attribution=['grad_shap'])

            attribution = output.get_primary_attributions(attribution_method)
            raw_prediction_tokens = output.tokens
            saved_json["attribution"] = attribution
            saved_json["tokens"] = raw_prediction_tokens

            with open(exp_saved_dir + str(i) + ".pkl", 'wb') as f:
                pickle.dump(saved_json, f)



def runing_exp_analysis():
    model_names = ["microsoft/codereviewer", "Salesforce/codet5p-220m", "Salesforce/codet5-base"]
    saved_model_names = ["codereviewer", "codet5p-220m", "codet5"]

    ## Tufano_et_al
    data_types = ["android", "google","ovirt"]
    data_sizes = ["small", "medium"]

    output_dir = "/path/to/output/"
    data_dir = "/path/to/Tufano_et_al/data/"
    for data_type in data_types:
        for data_size in data_sizes:
            for model_name, saved_model_name in zip(model_names, saved_model_names):
                checkpoint_prefix = 'checkpoint-best-loss/model.bin'
                model_path = output_dir + data_type + "/" + data_size + "/" + saved_model_name + "/" + checkpoint_prefix
                exp_saved_dir = output_dir + data_type + "/" + data_size + "/" + saved_model_name + "/new_exps/"
                if not os.path.exists(exp_saved_dir):
                    os.makedirs(exp_saved_dir)

                test_data_file = data_dir + data_type + "/" + data_size + "/"
                test_file_source_p = test_data_file + "test.code_before.txt" 
                test_file_target_p = test_data_file + "test.code_after.txt"
                
                with open(test_file_source_p, 'r') as f:
                    test_source_lines = f.readlines()
                with open(test_file_target_p, 'r') as f:
                    test_target_lines = f.readlines()   
                runing_explaining_by_grad_shap(model_name, model_path, test_source_lines, test_target_lines, exp_saved_dir)     


    ## code refinement Bugs2Fix
    data_sizes = ["small", "medium"]
    output_dir = "/path/to/Bugs2Fix/output/"
    data_dir = "/path/to/Bugs2Fix/data/"
    for data_size in data_sizes:
        for model_name, saved_model_name in zip(model_names, saved_model_names):
            checkpoint_prefix = 'checkpoint-best-loss/model.bin'
            model_path = output_dir + data_size + "/" + saved_model_name + "/" + checkpoint_prefix
            exp_saved_dir = output_dir + data_size + "/" + saved_model_name + "/new_exps/"
            if not os.path.exists(exp_saved_dir):
                os.makedirs(exp_saved_dir)

            test_data_file = data_dir + data_size + "/"
            test_file_source_p = test_data_file + "test.buggy-fixed.buggy" 
            test_file_target_p = test_data_file + "test.buggy-fixed.fixed"
            
            with open(test_file_source_p, 'r') as f:
                test_source_lines = f.readlines()
            with open(test_file_target_p, 'r') as f:
                test_target_lines = f.readlines()   
            runing_explaining_by_grad_shap(model_name, model_path, test_source_lines, test_target_lines, exp_saved_dir)



    # ## code transfer 
    data_types = ["java_cs", "cs_java"]
    output_dir = "/path/to/CodeTrans_Dataset/output/"
    data_dir = "/path/to/CodeTrans_Dataset/data/"
    for data_type in data_types:
        for model_name, saved_model_name in zip(model_names, saved_model_names):
            checkpoint_prefix = 'checkpoint-best-loss/model.bin'
            model_path = output_dir + data_type + "/" + saved_model_name + "/" + checkpoint_prefix
            exp_saved_dir = output_dir + data_type + "/" + saved_model_name + "/new_exps/"
            if not os.path.exists(exp_saved_dir):
                os.makedirs(exp_saved_dir)

            test_data_file = data_dir 
            if data_type == "java_cs":
                test_file_source_p = test_data_file + "test.java-cs.txt.java"
                test_file_target_p = test_data_file + "test.java-cs.txt.cs"
            else:
                test_file_source_p = test_data_file + "test.java-cs.txt.cs"
                test_file_target_p = test_data_file + "test.java-cs.txt.java"
                
            with open(test_file_source_p, 'r') as f:
                test_source_lines = f.readlines()
            with open(test_file_target_p, 'r') as f:
                test_target_lines = f.readlines()
            
            runing_explaining_by_grad_shap(model_name, model_path, test_source_lines, test_target_lines, exp_saved_dir)


    ## code generation CONCODE
    output_dir = "/path/to/CONCODE/output/"
    data_dir = "/path/to/CONCODE/data/"
    for model_name, saved_model_name in zip(model_names, saved_model_names):
        checkpoint_prefix = 'checkpoint-best-ppl/pytorch_model.bin'
        model_path = output_dir + saved_model_name + "/" + checkpoint_prefix
        exp_saved_dir = output_dir + saved_model_name + "/new_exps/"
        if not os.path.exists(exp_saved_dir):
            os.makedirs(exp_saved_dir)
        test_data_file = data_dir+"test.json"
        test_source_lines, test_target_lines = [], []
        with open(test_data_file, encoding="utf-8") as f:
            for line in f:     
                line = line.strip()
                js = json.loads(line)
                code = js["code"].replace("\n", " ")
                code=" ".join(code.strip().split())
                nl=js["nl"].replace("\n", " ")
                nl=" ".join(nl.strip().split())
                test_source_lines.append(nl)
                test_target_lines.append(code)

        runing_explaining_by_grad_shap(model_name, model_path, test_source_lines, test_target_lines, exp_saved_dir)


    ## CodeReview
    output_dir = "/path/to/CodeReview/output/"
    data_dir = "/path/to/CodeReview/data/coderefinement/"
    for model_name, saved_model_name in zip(model_names, saved_model_names):
        checkpoint_prefix = 'checkpoint-best-loss/model.bin'
        model_path = output_dir + saved_model_name + "/" + checkpoint_prefix
        exp_saved_dir = output_dir + saved_model_name + "/new_exps/"
        if not os.path.exists(exp_saved_dir):
            os.makedirs(exp_saved_dir)
        
        test_data_file = data_dir + "ref-test.jsonl"
        test_source_lines, test_target_lines = [], []
        items = [json.loads(line) for line in open(test_data_file)]
        for i in range(len(items)):
            item = items[i]            
            oldlines = item["old"].split("\n")
            newlines = item["new"].split("\n")
        
            oldlines = [line[1:].strip() for line in oldlines]
            newlines = [line[1:].strip() for line in newlines]
            oldlines = "\n".join(oldlines)
            newlines = "\n".join(newlines)
            oldlines = "<add>" + oldlines.replace("\n", "<add>")
            newlines = "<add>" + newlines.replace("\n", "<add>")
            comment = item["comment"]
            test_target_lines.append(newlines)
            oldlines = oldlines + comment
            test_source_lines.append(oldlines)
            
        runing_explaining_by_grad_shap(model_name, model_path, test_source_lines, test_target_lines, exp_saved_dir)
 

if __name__ == '__main__':
    runing_exp_analysis()