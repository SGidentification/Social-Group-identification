from vllm import LLM, SamplingParams
from time import time
import torch
import csv
from pathlib import Path
import gc
import os
import argparse
import timeit
import numpy as np
import torch.nn as nn
from transformers import ( AutoTokenizer, AutoModelForCausalLM)
from awq import AutoAWQForCausalLM
#from livelossplot import PlotLosses # pip install livelossplot
import pandas as pd
from huggingface_hub import snapshot_download
#from sklearn.metrics import classification_report
from seqeval.metrics import classification_report
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
# Set the random seed 
np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)
import wandb
import pprint
wandb.login()
import ray
import re
count_empty = 0
count_empty_after_jailbreaking = 0
sweep_config = {
    'method': 'grid',
    "name": "seqeval_wb",
    "metric": {"name": "f1", "goal": "maximize"},
    }
parameters_dict = {



    'temperature': {
        'values': [0.5]
        },
    'top_p': {
          'values': [1]
        },
     'top_k':{
        'values':[-1]},
    
    'presence_penalty':{
        'values':[-2]
    },
        'repetition_penalty':{
        'values':[0.1]
    },
    
        'prompt': {'values':[
            """Determine whether the text below contains mentions of social group. First, identify social groups based on this definition: Social groups are defined by two or more individuals who share some common characteristics. Commonalities shared by a social group include, for example, race, nationality, ethnicity, religion, gender, sexual orientation, socio-economic status, migration status, profession, family ties, and organizational and institutional ties. Second, do annotation for social groups starting with @@ and ending with ##.  The answer is in the followwing format:
The output starts with <annotation> and ends with </annotation> and number of words in the output are equal to the number of words in the input.  

Below are some examples:

Text: Because when the middle class does well , the poor have a ladder up and the wealthy still do very well . 
Answer: <annotation> Because when @@the middle class## does well , @@the poor## have a ladder up and @@the wealthy## still do very well . </annotation>
Reason: the middle class, the poor, and the wealthy each one has a size of the group more than one and each group shares some characteristics.



Text: hillary is stating publicly that her true allegiance is to the ultra-zionists . not the democratic party , not the usa , not anyone . 
Answer: <annotation> hillary is stating publicly that her true allegiance is to the @@ultra-zionists## . not the @@democratic party## , not @@the usa## , not anyone . </annotation> 
Reason: since the ultra-zionists, the democratic party, and the USA each one has a size of the group more than one, and each group shares some characteristics.


Text: President Biden said: near-record unemployment for Black and Hispanic workers . 
Answer: <annotation> President Biden said: near-record unemployment for @@Black and Hispanic workers## . </annotation>
Reason: Black and Hispanic workers is defined as a social group since it has a size of the group more than one, and each group shares some characteristics.

Text: Mr. Biden thanked Mr. Chaves , who was elected last year , for his leadership 
Answer: <annotation> Mr. Biden thanked Mr. Chaves , who was elected last year , for his leadership </annotation>
Reason: There is no mention of social groups in the text

Text:  """
]} ,
    
    }




sweep_config['parameters'] = parameters_dict
pprint.pprint(sweep_config)
sweep_id = wandb.sweep(sweep_config, project="test_final_socialgroups")



def read_csv(csv_file_path, col):
    df = pd.read_csv(csv_file_path, keep_default_na=False, encoding='latin-1',sep='\t')
    # Drop rows where the specified column has empty values
    df = df[df[col] != '']
    comments = df[col].tolist()
    return comments


def write_csv(csv_file_path, col, output):
    row_dict = {'Comment': '', 'Sentence annotation based on annotation aggrement': '', 'Raw Output': '', 'Output': ''}
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames = list(row_dict.keys()),delimiter='\t')
        if csv_file.tell() == 0:
            csv_writer.writeheader()
        row_dict[col] = output
        csv_writer.writerow(row_dict)
        

def extract_annotation(generated_text):
    
    generated_text = generated_text.lower()
    patterns = [
        r'(?<=<annotation>)(.*?)(?=</annotation>)',
        r'(?<=<annotation>)(.*?)(?=\n)',
        r'(?<=<annotation>)(.*?)(?=<\/annotation>)'
    ]

    for pattern in patterns:
        match = re.search(pattern, generated_text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return "EMPTY"


def postprocessing(seq_array): #input is an array of a list
    # Convert NumPy array to a Python list of strings
    seq_list = seq_array.astype(str)

    # Additional specific replacements using vectorized operations
    replacements = {
        "u2019": "",
        "u201c": "",
        "u201d": "",
        " 's": "'s",
        "s '": "s'",
        " n't": "n't",
        " 're": "'re",
        " 'd": "'d",
        " ##": "##",
        "@@ ": "@@",
        "\u2019": "",
        "\u201c": "",
        
        
    }
    for old, new in replacements.items():
        seq_list = np.core.defchararray.replace(seq_list, old, new)
    characters_to_remove = np.array(['.', ',', '"', '``', "'", '`', '?', '!', ":", ";", '/', '//', '\\'])
    # Create a translation table to remove specified characters
    translation_table = str.maketrans('', '', ''.join(characters_to_remove))
    # Apply translation to each element in the array using vectorized operations
    seq_list = np.core.defchararray.translate(seq_list, translation_table)
    eq_list = np.array([' '.join(s.split()) for s in seq_list])

    return eq_list

        
def convert_to_bio_format(seq):
    # Apply postprocessing to each element in the array
    bio_sequence = (len(seq))*["O"]
    inside_entity = False
    # Find the start and end indices of the gold entity
    for i, token in enumerate(seq):
        if token.startswith("@@") and token.endswith("##"):
            bio_sequence[i] = "B"
            inside_entity = False
        elif token.startswith("@@"):
            bio_sequence[i] = "B"
            inside_entity = False
            next_at_index = -1
            next_hash_index = -1
    
            # Find the index of the next token starting with @@
            for j, t in enumerate(seq[i + 1:]):
                if t.startswith("@@"):
                    next_at_index = i + 1 + j
                    break
    
            # Find the index of the next token ending with ##
            for j, t in enumerate(seq[i + 1:]):
                if t.endswith("##"):
                    next_hash_index = i + 1 + j
                    break
    
            if next_at_index != -1 and next_hash_index != -1 and next_hash_index < next_at_index:
                inside_entity = True
            if next_at_index == -1 and next_hash_index != -1:
                inside_entity = True
                
        elif inside_entity:
            bio_sequence[i] = "I"
            if not token.endswith("##"):
                inside_entity = True
            else:
                inside_entity = False
    return bio_sequence


def inference(args, config, llm):
    if os.path.isfile(args.csv_file_output):
        os.remove(args.csv_file_output)

    comments = read_csv(args.csv_file, 'body_cleaned')
    prompted_comments = [f"{config.prompt}{comment}" for comment in comments if len(comment) > 0 ]
    print(comments)
    batch_size = 408
    for i in range(0, len(comments), batch_size):
        batch_comments = prompted_comments[i:i + batch_size]
        if i + 1 == len(comments):
            continue
        t = time()
        sampling_params = SamplingParams(temperature=config.temperature, top_p=config.top_p, top_k = config.top_k, presence_penalty= config.presence_penalty,  repetition_penalty= config.repetition_penalty, max_tokens = len(max(comments, key=len)))
        batch_outputs = llm.generate(batch_comments, sampling_params)
        for j, output in enumerate(batch_outputs):
            generated_text = output.outputs[0].text
            generated_text_ann = extract_annotation(generated_text)
            write_csv(args.csv_file_output, 'Output', generated_text_ann)


def evaluation(config=None):

    #llm = LLM(model="TheBloke/Llama-2-70B-Chat-fp16", tensor_parallel_size=torch.cuda.device_count())
    llm = LLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1", tensor_parallel_size=torch.cuda.device_count())
    with wandb.init(config=config,tags=["ukpolitics"]): # ukpolitics, worldpolitics, economics, libertarian, politics
        config = wandb.config
        inference(args, config, llm)
        label = read_csv(args.csv_file, 'gold_annotation')
        pre = read_csv(args.csv_file_output, 'Output')
        label = postprocessing(np.array(label))
        pre = postprocessing(np.array(pre))
    
        prediction_list = []
        label_list =[]
        count = 0
        count_empty=0
        for l, p in zip(label, pre):
            if len(l.split(" ")) == len(p.split(" ")):
                label_list.append(convert_to_bio_format(l.split()))
                prediction_list.append(convert_to_bio_format(p.split()))
            else:
                if len(p.split(" ")) == 1:
                    count_empty += 1
                else:
                    count += 1

        wandb.log({"f1": f1_score(label_list, prediction_list,  mode='strict', scheme=IOB2, average= 'micro')})
        wandb.log({"recall": recall_score(label_list, prediction_list,  mode='strict', scheme=IOB2, average= 'micro')})
        wandb.log({"precision": precision_score(label_list, prediction_list,  mode='strict', scheme=IOB2, average= 'micro')})
        wandb.log({"count_empty": count_empty})
        wandb.log({"count_diff_size":count})
        wandb.log({"model":"mixtral"})
        ray.shutdown()
        


if __name__ == "__main__":
    
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Hello this space belogs to Farane!')
    parser.add_argument('--csv_file', required=True, help='Path to the input CSV file')
    parser.add_argument('--csv_file_output', required=True, help='Path to the output CSV file')
    global args
    args = parser.parse_args()
    wandb.agent(sweep_id, evaluation, count=100)
    pd.read_csv (args.csv_file_output,sep='\t').to_excel (args.csv_file_output+'output.xlsx', index = None, header=True) 

