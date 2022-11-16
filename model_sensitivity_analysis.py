import numpy as np
import os
from data_duplication_analysis import compute_bleu
import json

def remove_special_tokens(code_str):
    remove_lists = ["\n", "<add>", "<pad>", "<s>", "</s>", "<unk>", "{", "}", "(", ")", "[", "]", "<", ">", ":", ";", ",", ".", "=", "+", "-", "*", "/", "|", "&", "^", "%", "$", "#", "@", "!", "~", "`", "?", " "]
    for remove_q in remove_lists:
        code_str = code_str.replace(remove_q, "")
    return code_str


def computing_sensetive(test_source_file_p, test_target_file_p, dir_p, dirs_models):    
    """_summary_

    Args:
        test_source_file_p (str): path of test source file
        test_target_file_p (str): path of test target file
        dir_p (str): directory of model prediction
        dirs_models (list): list of model names
    """
    test_sources = [line.strip() for line in open(test_source_file_p, 'r').readlines()]
    test_targets = [line.strip() for line in open(test_target_file_p, 'r').readlines()]
            
    for model in dirs_models:
        model_p = dir_p + model + "/"
        
        pred_file_p = model_p + "predictions.txt"
        pre_results_p = model_p + "test_result.txt"

        
        if os.path.exists(pred_file_p):            
            preds= [line.strip() for line in open(pred_file_p, 'r').readlines()]
            results = [int(line.strip()) for line in open(pre_results_p, 'r').readlines()]
            target_after_remove = []
            pred_after_remove =[]
            
            unchanged = []
            for i in range(len(preds)):
                if test_sources[i].replace(" ","") == preds[i].replace(" ",""):
                    unchanged.append(True)
                else:
                    unchanged.append(False)
                    
            unchanged_after_remove_special = []
            for i in range(len(preds)):
                if remove_special_tokens(test_sources[i]) == remove_special_tokens(preds[i]):
                    unchanged_after_remove_special.append(True)
                else:
                    unchanged_after_remove_special.append(False)
                    target_after_remove.append([test_targets[i]])
                    pred_after_remove.append(preds[i])
            
                    
            dict_pred_count = dict()
            for i in range(len(preds)):
                if preds[i].replace(" ","") in dict_pred_count.keys():
                    dict_pred_count[preds[i].replace(" ","")] += 1
                else:
                    dict_pred_count[preds[i].replace(" ","")] = 1
            
            not_same_count = 0
            for key in dict_pred_count.keys():
                if dict_pred_count[key] == 1:
                    not_same_count += 1
                
            # print("accuracy", np.sum(results)/len(results), np.sum(results), len(results))
            # print("unchanged rate",np.sum(unchanged)/len(unchanged),np.sum(unchanged), len(unchanged))
            # print("unchanged rate after remove special",np.sum(unchanged_after_remove_special)/len(unchanged_after_remove_special),np.sum(unchanged_after_remove_special), len(unchanged_after_remove_special))
                        
            test_targets_lists = [[line] for line in test_targets]
            old_bleu, _, _, _, _, _  = compute_bleu(test_targets_lists, preds)
            new_bleu, _, _, _, _, _  = compute_bleu(target_after_remove,  pred_after_remove)
            
            print(np.sum(results)/len(results), np.sum(unchanged)/len(unchanged),np.sum(unchanged_after_remove_special)/len(unchanged_after_remove_special),old_bleu, new_bleu)
    

def sensetivity_analysis():
    source_dir = "icse19/data/"    
    prediction_dir = "icse19/output/"
    datatypes = ["android", "google", "ovirt"]
    datasizes = ["small", "medium"]
    
    for datatype in datatypes:
        for datasize in datasizes:
            dir_p = prediction_dir + datatype + "/" + datasize + "/"
            
            dirs_models = ["codetrans"]
            #dirs_models = os.listdir(dir_p)

            test_source_file_p = source_dir + datatype + "/" + datasize + "/test.code_before.txt"
            test_target_file_p = source_dir + datatype + "/" + datasize + "/test.code_after.txt"
            computing_sensetive(test_source_file_p, test_target_file_p, dir_p, dirs_models) 
    

if __name__ == "__main__":
    sensetivity_analysis()