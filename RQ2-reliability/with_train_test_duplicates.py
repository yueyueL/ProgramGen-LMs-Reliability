import numpy as np
import os
import collections
import math
import json
from util import compute_bleu
def tokenize(s):
    return s.split()


def compare_similarity_for_tests(test_sources, train_sources, saved_path):
    overall_dump_results, max_indexs = [], []

    for i in range(len(test_sources)):
        similrity = []
        for j in range(len(train_sources)):
            reference, translation = [tokenize(test_sources[i])], tokenize(train_sources[j])
            bleu_score, _, _, _, _, _  = compute_bleu([reference], [translation])
            similrity.append(bleu_score)

        max_similarity = np.max(similrity)
        max_index = np.argmax(similrity)
        max_list = [max_similarity.item(), max_index.item()] if isinstance(max_index, np.integer) else [max_similarity, max_index]
        overall_dump_results.append(max_list)

    with open(saved_path, 'w') as f:
        json.dump(overall_dump_results, f)


def saved_correctness(dir_p, test_source_file_p,train_source_file_p, saved_path):
    
    if "CONCODE" in dir_p:
        test_source, train_source = [], []
        with open(test_source_file_p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                js =json.loads(line)
                code = js["code"].replace("\n", " ")
                code=' '.join(code.strip().split())
                nl = js["nl"].replace("\n", " ")
                nl=' '.join(nl.strip().split())
                test_source.append(code)
        with open(train_source_file_p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                js =json.loads(line)
                code = js["code"].replace("\n", " ")
                code=' '.join(code.strip().split())
                nl = js["nl"].replace("\n", " ")
                nl=' '.join(nl.strip().split())
                train_source.append(code)
    elif "CodeReview" in dir_p:
        train_source, test_source = [], []
        items = [json.loads(line) for line in open(test_source_file_p)]
        train_items = [json.loads(line) for line in open(train_source_file_p)]
        for i in range(len(items)):
            item = items[i]
            newlines = item["new"].split("\n")
            newlines = [line[1:].strip() for line in newlines]
            newlines = "\n".join(newlines)
            newlines = "<add>" + newlines.replace("\n", "<add>")

            oldlines = item["old"].split("\n")
            oldlines = [line[1:].strip() for line in oldlines]
            oldlines = "\n".join(oldlines)
            oldlines = "<add>" + oldlines.replace("\n", "<add>")
            comment = item["comment"]
            oldlines = oldlines + comment
            test_source.append(newlines)
        for i in range(len(train_items)):
            item = train_items[i]
            newlines = item["new"].split("\n")
            newlines = [line[1:].strip() for line in newlines]
            newlines = "\n".join(newlines)
            newlines = "<add>" + newlines.replace("\n", "<add>")

            oldlines = item["old"].split("\n")
            oldlines = [line[1:].strip() for line in oldlines]
            oldlines = "\n".join(oldlines)
            oldlines = "<add>" + oldlines.replace("\n", "<add>")
            comment = item["comment"]
            oldlines = oldlines + comment
            train_source.append(newlines)
    else:
        train_source = [line.strip() for line in open(train_source_file_p, 'r').readlines()]
        test_source = [line.strip() for line in open(test_source_file_p, 'r').readlines()]

    compare_similarity_for_tests(test_source, saved_path)

def save_similarity_model_results():

    ################ Tufano et al. #################

    prediction_dir = "/path/to/tufano_et_al/output/"
    data_test_dir = "/path/to/tufano_et_al//data/"
    datatypes = ["android", "google", "ovirt"]
    datasizes = ["small", "medium"]    
    for datatype in datatypes:    
        for datasize in datasizes:
            print(datatype, datasize)
            dir_p = prediction_dir + datatype + "/"  + datasize + "/"
            test_source_file_p = data_test_dir+ datatype + "/"    + datasize + "/test.code_after.txt"
            train_source_file_p = data_test_dir+ datatype + "/"    + datasize + "/train.code_after.txt"
            saved_path = "/path/to/similarity/within_test_similarity/tufano_"+ datatype +"_"+ datasize +"_counts.json"
            saved_correctness(dir_p, test_source_file_p,train_source_file_p, saved_path)
    
    ## Bugs2Fix
    source_dir = "/path/to/Bugs2Fix/data/"
    prediction_dir = "/path/to/Bugs2Fix/output/"
    datasizes = ["small", "medium"]
    for datasize in datasizes:
        dir_p = prediction_dir  + datasize + "/"
        test_source_file_p = source_dir  + datasize + "/test.buggy-fixed.fixed"
        train_source_file_p = source_dir  + datasize + "/train.buggy-fixed.fixed"      
        saved_path = "/path/to/similarity/within_test_similarity/bug2fix_"+ datasize +"_counts.json"
        saved_correctness(dir_p, test_source_file_p,train_source_file_p, saved_path)

    ## CodeTransfer-Data
    source_dir = "/path/to/CodeTrans_Dataset/data/"
    prediction_dir = "/path/to/CodeTrans_Dataset/output/"
    datatypes = ["cs_java", "java_cs"]
    for datatype in datatypes:
        dir_p = prediction_dir  + datatype + "/"
        if datatype == "cs_java":
            test_source_file_p = source_dir + "test.java-cs.txt.java"
            train_source_file_p = source_dir + "train.java-cs.txt.java"
        else:
            test_source_file_p = source_dir  + "test.java-cs.txt.cs"
            train_source_file_p = source_dir  + "train.java-cs.txt.cs"
        saved_path = "/path/to/similarity/within_test_similarity/codetransfer_"+ datatype +"_counts.json"
        saved_correctness(dir_p, test_source_file_p,train_source_file_p, saved_path)


    ## code review
    source_dir = "/path/to/CodeReview/data/coderefinement/"
    prediction_dir = "/path/to/CodeReview/output/"
    
    dir_p = prediction_dir
    test_source_file_p = source_dir + "test.json"
    train_source_file_p = source_dir + "train.json"
    saved_path = "/path/to/similarity/within_test_similarity/codereviewer_counts.json"
    saved_correctness(dir_p, test_source_file_p,train_source_file_p, saved_path)


    ## code generation
    source_dir = "/path/to/CONCODE/data/"
    prediction_dir = "/path/to/CONCODE/output/"
    dir_p = prediction_dir
    test_source_file_p = source_dir + "test.json"
    train_source_file_p = source_dir + "train.json"
    saved_path = "/path/to/similarity/within_test_similarity/concode_counts.json"
    saved_correctness(dir_p, test_source_file_p,train_source_file_p, saved_path)

if __name__ == '__main__':
    save_similarity_model_results()