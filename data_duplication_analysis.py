import numpy as np
import os
import collections
import math
import json

def _get_ngrams(segment, max_order):
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1
    return ngram_counts

# compute BLEU score
def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, max_order+1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                        (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)

# compute the accuracy of the code generation
def calcalute_accuray(y_true_list, y_pred_list):
    account_correct = 0
    for i in range(len(y_true_list)):
        y_true = y_true_list[i].replace(' ', '')
        y_pred = y_pred_list[i].replace(' ', '')
        if y_true == y_pred:
            account_correct += 1
    return account_correct / len(y_true_list)


def compare_dumplication_lists(train_sources, train_targets, test_sources, test_targets):
    dump_results_sources, dump_results_tragets =[], []
    for i in range(len(test_sources)):
        is_dump_source, is_dump_target = False, False
        for j in range(len(train_sources)):
            if test_targets[i].replace(" ","") == train_targets[j].replace(" ",""):
                is_dump_target = True
            if test_sources[i].replace(" ","") == train_sources[j].replace(" ",""):
                is_dump_source = True
        dump_results_sources.append(is_dump_source)
        dump_results_tragets.append(is_dump_target)
    return dump_results_sources, dump_results_tragets

def simple_changes(sources, targets):
    simple_sources, simple_targets = [],[]
    for i in range(len(sources)):
        source_token = sources[i].split(" ")
        target_token = targets[i].split(" ")
        
        start_index, end_index = getChanged_tokens(source_token, target_token)
        start_index_source, end_index_source = get_start_and_end_index(source_token, start_index, end_index)
        start_index_target, end_index_target = get_start_and_end_index(target_token, start_index, end_index)        
        simple_sources.append(" ".join(source_token[start_index_source:end_index_source]))
        simple_targets.append(" ".join(target_token[start_index_target:end_index_target]))
    return simple_sources, simple_targets
              
def getChanged_tokens(source_tokens, target_tokens):
    change_start_index, change_end_index = 0, 0
    i = 0
    while i < len(source_tokens) and i < len(target_tokens):
        change_start_index = i
        if source_tokens[i] != target_tokens[i]:
            break
        i += 1

    i = -1
    while i >= -len(source_tokens) and i >= -len(target_tokens):
        change_end_index = i
        if source_tokens[i] != target_tokens[i]:
            break
        i -= 1
    return change_start_index, change_end_index

def get_start_and_end_index(tokens, start_index, end_index):
    if start_index > len(tokens)+end_index:
        return start_index, end_index+len(tokens)
    else: 
        return start_index, end_index+len(tokens)+1   
    

def compare_dumplication_for_changes(simple_train_sources,simple_train_targets, simple_test_sources, simple_test_targets):
    overall_dump_results= []
    for i in range(len(simple_test_sources)):
        is_dump = False
        for j in range(len(simple_train_sources)):
            if simple_test_sources[i] == simple_train_sources[j] and simple_test_targets[i] == simple_train_targets[j]:
                is_dump = True
                break
        overall_dump_results.append(is_dump)
    return overall_dump_results


def read_codereviewer_json(file_path):
    items =[json.loads(line) for line in open(file_path, 'r')]
    test_source_lines =[]
    test_target_lines =[]
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
        source = oldlines
        target = newlines
        if len(source) == 0 or len(target) == 0:
            continue
        
        test_source_lines.append(source)
        test_target_lines.append(target)    
        
    return test_source_lines, test_target_lines            

def computing_duplication_for_codereviewer(train_file_p, test_file_p,  pred_dir, dirs_models):
    """_summary_

    Args:
        train_file_p (str): path of the train file
        test_file_p (str): path of the test file
        pred_dir (str): directory of the prediction file
        dirs_models (list): list of the directories of the studied models
    """
    test_sources,test_targets = read_codereviewer_json(test_file_p)
    train_sources,train_targets = read_codereviewer_json(train_file_p)
    simple_train_sources, simple_train_targets = simple_changes(train_sources, train_targets)  
    
    simple_test_sources, simple_test_targets = simple_changes(test_sources, test_targets)
    
    overall_dump_results =compare_dumplication_for_changes(simple_train_sources, simple_train_targets, simple_test_sources, simple_test_targets)
            
    for model in dirs_models:
        model_p = pred_dir + model + "/"
        print(model_p)
        #read test result
        pred_file_p = model_p + "predictions.txt"
        pre_results_p = model_p + "test_result.txt"

        
        if os.path.exists(pred_file_p):

            
            preds= [line.strip() for line in open(pred_file_p, 'r').readlines()]
            preds_removed_dumplication = []
            groud_truths_removed_dumplication = []
            results = [int(line.strip()) for line in open(pre_results_p, 'r').readlines()]
            
            count_right, count_false = 0, 0
            
            for i in range(len(results)):
                if overall_dump_results[i] == True:
                    if results[i] == 1:
                        count_right += 1
                    else:
                        count_false += 1
                else:
                    preds_removed_dumplication.append(preds[i])
                    groud_truths_removed_dumplication.append(test_targets[i])

            print("dumplicate_rate:", np.sum(overall_dump_results)/len(results))
            print("accuracy:", np.sum(results)/len(results)) 
            print("new_accuracy_after_dumplication", (np.sum(results) - count_right)/(len(results)-np.sum(overall_dump_results)))                
            test_targets_lists = [[line] for line in test_targets]
            groud_truths_removed_dumplication = [[line] for line in groud_truths_removed_dumplication]
            original_bleu, _, _, _, _, _ = compute_bleu(test_targets_lists, preds)
            new_bleu, _, _, _, _, _ = compute_bleu(groud_truths_removed_dumplication, preds_removed_dumplication)
            
            print("original_bleu:", original_bleu)
            print("new_bleu:", new_bleu)
            print()
    
def computing_duplication(train_source_file_p, train_target_file_p, test_source_file_p, test_target_file_p, pred_dir, dirs_models):
    """_summary_

    Args:
        train_file_p (str): directory of the train file
        test_file_p (str): directory of the test file
        pred_dir (str): directory of the prediction file
        dirs_models (list): list of the directories of the studied models
    """    

    test_sources = [line.strip() for line in open(test_source_file_p, 'r').readlines()]
    test_targets = [line.strip() for line in open(test_target_file_p, 'r').readlines()]
    
    train_sources = [line.strip() for line in open(train_source_file_p, 'r').readlines()]
    train_targets = [line.strip() for line in open(train_target_file_p, 'r').readlines()]    
    
    simple_train_sources, simple_train_targets = simple_changes(train_sources, train_targets)  
    
    simple_test_sources, simple_test_targets = simple_changes(test_sources, test_targets)
    
    overall_dump_results =compare_dumplication_for_changes(simple_train_sources, simple_train_targets, simple_test_sources, simple_test_targets)
    
            
    for model in dirs_models:
        model_p = pred_dir + model + "/"
        print(model_p)
        #read test result
        pred_file_p = model_p + "predictions.txt"
        pre_results_p = model_p + "test_result.txt"

        
        if os.path.exists(pred_file_p):

            
            preds= [line.strip() for line in open(pred_file_p, 'r').readlines()]
            preds_removed_dumplication = []
            groud_truths_removed_dumplication = []
            results = [int(line.strip()) for line in open(pre_results_p, 'r').readlines()]
            
            count_right, count_false = 0, 0
            
            for i in range(len(results)):
                if overall_dump_results[i] == True:
                    if results[i] == 1:
                        count_right += 1
                    else:
                        count_false += 1
                else:
                    preds_removed_dumplication.append(preds[i])
                    groud_truths_removed_dumplication.append(test_targets[i])

            print("dumplicate_rate:", np.sum(overall_dump_results)/len(results))
            print("accuracy:", np.sum(results)/len(results)) 
            print("new_accuracy_after_dumplication", (np.sum(results) - count_right)/(len(results)-np.sum(overall_dump_results)))                
            test_targets_lists = [[line] for line in test_targets]
            groud_truths_removed_dumplication = [[line] for line in groud_truths_removed_dumplication]
            original_bleu, _, _, _, _, _ = compute_bleu(test_targets_lists, preds)
            new_bleu, _, _, _, _, _ = compute_bleu(groud_truths_removed_dumplication, preds_removed_dumplication)
            
            print("original_bleu:", original_bleu)
            print("new_bleu:", new_bleu)
            print()

if __name__ == "__main__":
    
    ### data duplication analysis for ICSE19 dataset
    source_dir = "icse19/data/"
    prediction_dir = "icse19/output/"
    datatypes = ["android", "google", "ovirt"]
    datasizes = ["small", "medium"]
    
    for datatype in datatypes:    
        for datasize in datasizes:
            dir_p = prediction_dir + datatype + "/"  + datasize + "/"
            dirs_models = ["t5", "codereviewer"]

            test_source_file_p = source_dir+ datatype + "/"    + datasize + "/test.code_before.txt"
            test_target_file_p = source_dir+ datatype + "/"    + datasize + "/test.code_after.txt"
            
            train_source_file_p = source_dir+ datatype + "/"    + datasize + "/train.code_before.txt"
            train_target_file_p = source_dir+ datatype + "/"    + datasize + "/train.code_after.txt"
            
            computing_duplication(train_source_file_p, train_target_file_p, test_source_file_p, test_target_file_p, dir_p, dirs_models)    
    
    ### data duplication analysis for TOSEM19 dataset
    source_dir = "tosem19/data/"
    prediction_dir = "tosem19/output/" 
    datasizes = ["small", "medium"]
    
    for datasize in datasizes:
        dir_p = prediction_dir  + datasize + "/"
        dirs_models = os.listdir(dir_p)
        dirs_models = ["t5", "codereviewer"]

        test_source_file_p = source_dir  + datasize + "/test.buggy-fixed.buggy"
        test_target_file_p = source_dir  + datasize + "/test.buggy-fixed.fixed"
        
        train_source_file_p = source_dir  + datasize + "/train.buggy-fixed.buggy"
        train_target_file_p = source_dir  + datasize + "/train.buggy-fixed.fixed"
        
        computing_duplication(train_source_file_p, train_target_file_p, test_source_file_p, test_target_file_p, dir_p, dirs_models)
            
    ### data duplication analysis for NIPS21 datasets
    source_dir = "nips21/data/"
    prediction_dir = "nips21/output/cs_java/"    
    dir_p = prediction_dir + "/"
    dirs_models = [ "t5", "codereviewer"]
    test_source_file_p = source_dir+ "/test.java-cs.txt.cs"
    test_target_file_p = source_dir+  "/test.java-cs.txt.java"
    
    train_source_file_p = source_dir+  "/train.java-cs.txt.cs"
    train_target_file_p = source_dir+  "/train.java-cs.txt.java"
    
    computing_duplication(train_source_file_p, train_target_file_p, test_source_file_p, test_target_file_p, dir_p, dirs_models)        
    
    #### data duplication analysis for FSE22 dataset
    source_dir = "fse22/Code_Refinement/"
    prediction_dir = "fse22/output/"    
    dir_p = prediction_dir + "/"
    dirs_models = ["t5", "codereviewer"]
    test_file_p = source_dir+  "/ref-test.jsonl"
    train_file_p = source_dir+  "/ref-train.jsonl"
    
    computing_duplication_for_codereviewer(train_file_p, test_file_p, dir_p, dirs_models)