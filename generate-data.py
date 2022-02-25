# this python script generates data from GPT2

# system design - 
# input = Role dataset few shot samples - random 4 or 8 - get from data/raw_data
# output = More samples like them - generated_data/role-880-date-version
# system = GPT2

from os import truncate
from unittest.util import _MAX_LENGTH
from  transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import random
import argparse

def get_arguments():
    """
    Get arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='neg', help="'neg' for neg-simp, 'role' for role-88")
    args = parser.parse_args()
    return args

def generate_role88():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    model = GPT2LMHeadModel.from_pretrained('gpt2-xl', pad_token_id = tokenizer.eos_token_id)

    sentence = """FGoal is to reverse the role in the sentence pair.
    Generate more sentence pair like these. 

    the journalist investigated which athlete the team had recruited,
    the journalist investigated which team the athlete had joined,

    the detective interviewed which witness the doctor had suspected,
    the detective interviewed which doctor the witness had seen,

    the teacher lectured which student the class had ignored,
    the teacher lectured which class the student had left,

    the police reported which criminal the witness had described,
    the police reported which witness the criminal had robbed,
    """
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    output = model.generate(input_ids, max_length=200, do_sample=True, temperature=2.0)
    return tokenizer, output


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
    if args.dataset == 'neg':
        generate_negsimp()
    elif args.dataset == 'role':
        tokenizer, output = generate_role88()
        print(tokenizer.decode(output[0]))

if __name__ == "__main__":
    main()