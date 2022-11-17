import json
from ecco import from_pretrained
import torch
import os
import numpy as np
import pickle
import pynvml


def attri_array_reshape(attri_array):
    new_attri = []
    for attri in attri_array:
        new_attri.append(attri[:len(attri_array[0])-1])
        
    return np.array(new_attri)

def group_tokens(tokens, attris, bpe_split='##'):
    """
    tokens: list of code tokens
    attris: list of attributions (feature importance) for each token
    bpe_split: the token that indicates a token is a subword of a word
    """
    new_tokens_list = []
    new_attrs_masks = []
    # ▁ 
    for i, token in enumerate(tokens):
        if token.startswith(bpe_split):
            new_tokens_list.append(token[1:])
            #print(new_tokens_list)
            new_attrs_masks.append(1)
        elif token=='<s>':
            new_tokens_list.append(token)
            new_attrs_masks.append(-1)
        else:
            new_tokens_list[-1] += token         
            if token=='</s>':
                new_attrs_masks.append(-1)
            else:
                new_attrs_masks.append(0)
            
    new_attrs = []
    for i, attr in enumerate(attris):
        if new_attrs_masks[i] == 1 or new_attrs_masks[i] == -1:
            start_i  = i + 1
            sum_attr = attr if new_attrs_masks[i] == 1 else 0
            count_attri = 1 if new_attrs_masks[i] == 1 else 0
            
            while start_i < len(new_attrs_masks) and new_attrs_masks[start_i] == 0:
                sum_attr += attris[start_i]
                start_i += 1
                count_attri += 1
            new_attrs.append(sum_attr/count_attri)
    
            if start_i == len(new_attrs_masks)-1:
                break
    new_tokens_list[0] =  new_tokens_list[0].replace('<s>', '')
    new_tokens_list[-1] =  new_tokens_list[-1].replace('</s>', '')      
    return new_tokens_list, new_attrs

def save_dict_exp(dict, path): 
    with open(path, 'wb') as f:
        pickle.dump(dict, f)

def main_code_updates():
    #pynvml.nvmlInit()
    #handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_names = ["Salesforce/codet5-base", "microsoft/codereviewer", "razent/cotext-1-ccg", "t5-base", "SEBIS/code_trans_t5_base_code_comment_generation_java"]
    # saved_model_names = ["codet5", "codereviewer", "cotext", "t5", "codetrans"]
    
    model_names = ["microsoft/codereviewer"]
    saved_model_names = ["codereviewer"]
    
    # data_types = ["android", "google","ovirt"]
    # data_sizes = ["small", "medium"]
    
    data_types = [ "android"]
    data_sizes = ["small"]
    
    for data_type in data_types:
        for data_size in data_sizes:
            for model_j in range(len(model_names)):
                model_name_or_path = model_names[model_j]
                
                output_dir ="code_updates/output/"+data_type+"/"+data_size+"/"+saved_model_names[model_j]+"/"
                model_name="model.bin"
                test_data_file = "code_updates/data/"+data_type+"/"+data_size+"/"
                exp_saved_dir = output_dir+"pred_exps/"
                
                checkpoint_prefix = 'checkpoint-best-loss/'+model_name
                output_model_dir = os.path.join(output_dir, '{}'.format(checkpoint_prefix)) 
                print(exp_saved_dir)
                if not os.path.exists(exp_saved_dir):
                    os.makedirs(exp_saved_dir) 


                model_config = {
                    'embedding': 'shared.weight',
                    'type': 'enc-dec',
                    'activations': ['wo'],
                    'token_prefix': '',
                    'partial_token_prefix': '##'
                    
                }
                
                model_config_bart = {
                    "type": "enc-dec",
                    "embedding": "shared",
                    'activations': ['fc2'],
                    "token_prefix": '',
                    "partial_token_prefix": '##'
                    
                }

                if model_name_or_path in ["Salesforce/codet5-base", "microsoft/codereviewer", "razent/cotext-1-ccg", "SEBIS/code_trans_t5_base_code_comment_generation_java"]:
                    lm = from_pretrained(model_name_or_path, local_model_path=output_model_dir, model_config = model_config, attention=True, device = device)
                elif model_name_or_path in ["uclanlp/plbart-base"]:
                    lm = from_pretrained(model_name_or_path, local_model_path=output_model_dir, model_config = model_config_bart, attention=True, device = device)        
                else:
                    lm = from_pretrained(model_name_or_path, local_model_path=output_model_dir, attention=True, device = device)
                
                test_file_source_p = test_data_file + "test.code_before.txt" 
                test_file_target_p = test_data_file + "test.code_after.txt"
                
                with open(test_file_source_p, 'r') as f:
                    test_source_lines = f.readlines()
                with open(test_file_target_p, 'r') as f:
                    test_target_lines = f.readlines()
                
                #attributions = ['ig', 'saliency', 'input_x_grad','grad_shap']
                attributions = ['grad_shap']
                
                
                split_bpe_token_dict = {"t5-base": '▁', "Salesforce/codet5-base": 'Ġ', "microsoft/codereviewer": 'Ġ', "uclanlp/plbart-base": '▁', "razent/cotext-1-ccg": '▁', "SEBIS/code_trans_t5_base_code_comment_generation_java": '▁'}

                for i in range(len(test_source_lines)):
                    source = test_source_lines[i].strip()
                    target = test_target_lines[i].strip()
                    torch.cuda.empty_cache()
                    # if os.path.exists(exp_saved_dir + "saliency" + "/" + str(i) + ".pl"):
                    #     continue
                    # if i <220:
                    #     continue
                    
                    source_ids = lm.tokenizer.encode(source)
                    raw_tokens = lm.tokenizer.convert_ids_to_tokens(source_ids)
                    if len(raw_tokens) < 64:
                        output = lm.generate(source, max_length= 1000, do_sample = False, attribution=attributions)
                        
                        for j, attribution_method in enumerate(attributions):
                                            
                            exp_pred_save_attri_dir = exp_saved_dir + attribution_method + "/"
                            if not os.path.exists(exp_pred_save_attri_dir):
                                os.makedirs(exp_pred_save_attri_dir)
                                
                            attri = output.get_primary_attributions(attribution_method)
                            # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            # print("memory used: ", meminfo.used/1024/1024)
                            attri = attri_array_reshape(attri)
                            avg_attri = np.mean(attri, axis=0)

                            dict_tokens_raw = [(raw_tokens[i], avg_attri[i]) for i in range(len(raw_tokens))]  
                            #save_dict_exp(dict_tokens_raw, exp_pred_save_attri_dir + str(i) + ".pl")
                            new_tokens, new_attrs = group_tokens(raw_tokens, avg_attri, split_bpe_token_dict[model_name_or_path])
                            dirct_token_group = [(new_tokens[i], new_attrs[i]) for i in range(len(new_attrs))]
                            save_dict_exp(dirct_token_group, exp_pred_save_attri_dir + str(i) + "_group.pl")
                        

if __name__ == '__main__':
    main_code_updates()
    