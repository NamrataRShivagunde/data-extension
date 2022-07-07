# this python script generates data from GPT2

# system design - 
# input = Role dataset few shot samples - random 4 or 8 - get from data/raw_data
# output = More samples like them - generated_data/role-880-date-version
# system = GPT2

from dis import Instruction
from os import truncate
from unittest.util import _MAX_LENGTH
from  transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import random
import argparse
import re
import os
import subprocess
import json


def get_arguments():
    """
    Get arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='neg', help="'neg' for neg-simp, 'role' for role-88")
    # parser.add_argument("--curl", default=None, help="curl command for gpt3")
    args = parser.parse_args()
    return args

def generate_role88():
    """
    generate role-88 pairs using raw role-88
    """
    data = []
    with open("data/raw_data/ROLE-88/ROLE-88.tsv","r") as f:
        data = f.readlines() # readlines() returns a list of items, each item is a line in your file
    NUM_SAMPLES = 4
    
    for i in range(1, len(data)):
        a = data[i][5:].replace("|","\t").strip().split('\t')
        a = a[0] + a[1] + ",\n"
        f = open("role-88-raw.txt", "a") 
        f.write(a)

    # for i in range(25): 
    #     print(i)
    #     random_num_list = []
    #     while len(random_num_list) < NUM_SAMPLES:
    #         random_num_list.append(random.randrange(1, len(data), 2))
    #         random_num_list == list(set(random_num_list)) # list of random integer between item 1 till len of data
    #     print(random_num_list)
    #     # create the prompt. choose sample pairs using the random_num_list
    #     prompt = "The task is to reverse the role in the sentences. Generate more sentences like this: "
    #     for item in random_num_list:
    #         sample1 = data[item][5:].replace("|","\t").strip().split('\t')
    #         sample1 = sample1[0] + sample1[1]
        
    #         sample2 = data[item+1][5:].replace("|","\t").strip().split('\t')
    #         sample2 = sample2[0] + sample2[1]

    #         random_samples = sample1 + ',' + sample2 + ','
    #         prompt = prompt + random_samples
 

    #     # pass this prompt to the curl command to gpt3
    # # TO DO remove the key
    #     curl_req = 'curl https://api.openai.com/v1/completions \
    #     -H "Content-Type: application/json" \
    #     -H "Authorization: Bearer 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" \
    #     -d \'{"model": "text-davinci-002", "prompt": ' + '"' + prompt + '"' +', "temperature": 0.64, "max_tokens": 100}\''
        
    #     print("prompt-----------------",prompt)
    #     # Getting resposnse from gpt3
    #     gpt3result = subprocess.check_output(curl_req, shell=True)
    #     # gpt3result = '{"id":"cmpl-5RE97W2MnajgO2Lln0eVDLp7W3pyS","object":"text_completion","created":1657170917,"model":"text-davinci-002","choices":[{"text":"\\n\\nThis is a test","index":0,"logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":5,"completion_tokens":6,"total_tokens":11}}\n'
    #     gpt3result = json.loads(gpt3result)

    #     # extracting the output text from gpt3 response
    #     gpt3result = gpt3result['choices'][0]['text']
    #     print(gpt3result)
        
    #     # dumping the extracted text to a txt file
    #     file = open("role-88-generated-notcleaned.txt", "a") 
    #     file.write(gpt3result)
    #     # clean it manually
    return 0,0


def generate_negsimp():
    """
    Generates the dataset NEG-SIMP using categories and subcategories from the original paper
    """
    file = pd.read_csv('data/neg-simp-categories.csv')
    cat_subcat = dict(zip(file['Category'], file['Subcategory']))
    
    sent = []
    for cat, subcat in cat_subcat.items():
        subcat = subcat.split(',')
        for item in subcat:
            aff = 'A ' + item + " is (a/an) " + cat.lower() + ','
            neg = 'A ' + item + " is not (a/an) " + random.choice([x.lower() for x in file['Category'] if x != cat]) + ','
            sent.append(aff)
            sent.append(neg)
    textfile = open("data/neg_simp_generated.txt", "w")
    for element in sent:
        textfile.write(element + "\n")
    textfile.close()


def main():
    args = get_arguments()
    if args.dataset == 'negsimp':
        generate_negsimp()
    elif args.dataset == 'role':
        tokenizer, output = generate_role88()
        # print(tokenizer.decode(output[0]))

if __name__ == "__main__":
    main()