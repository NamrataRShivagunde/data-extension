from dis import Instruction
from numpy import mod
import transformers
import torch
import argparse
import re
import subprocess
import json


def get_args():
    """ Set hyperparameters """
    parser = argparse.ArgumentParser()
  
    parser.add_argument("file_path", help="file path of the extended dataset e.g. data/neg_simp_generated.txt")
    parser.add_argument("model_name_or_path", help="Huggingface pretrained model name/path")
    parser.add_argument("--key", help="key for openai gpt3")

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()
    return args

def process_data(file_path, modeldir):
    """Convert the sentences into source and label

    Arguments 
        file_path (string) : file path
        modeldir (string) : model name
    
    Return
        source (list) : convert e.g.The librarian documented which journalist the celebrities had avoided, 
                                    The librarian documented which celebrities the journalist had interviewed, 

                            to
        
                            ['The librarian documented which journalist the celebrities had [MASK]',
                             'The librarian documented which celebrities the journalist had [MASK]']
        
        label (list) : e.g.     ['avoided', 'interviewed']
    """
    source = []
    label = []
    
    # Read line by line, remove (, ), , and replace last word with [MASK] and add to source list
    # Append target word to label list
    with open(file_path) as f:
        text = f.readlines()
        for line in text:
            line= line.replace(',', "")
            splitted_line = line.split(" ")
            line_label = splitted_line[-1]
            if modeldir.startswith("roberta"):
                line_source = line.replace(line_label, '<mask> .')
            elif modeldir.startswith("gpt"):
                line_source = line.replace(line_label, '')
            elif modeldir.startswith('t5'):
                line_source = line.replace(line_label, '<extra_id_0>.')
            else:
                line_source = line.replace(line_label, '[MASK].')
            label.append(line_label.rstrip("\n"))
            source.append(line_source)
    return source, label

def gpt3_responses(file_path, key):
    """
    Arguments
        file_path
        key : OPEN AI gpt 3 key
    
    Return
        top1pred [list] : top 5 predictions from gpt3 
        top5pred [list] : top 5 predictions from gpt3 . Note top 5 are not in decreaing order of prob thats 
                                why we have top1 different than first element of top5.
    """
    
    source, label = process_data(file_path, 'gpt')
    top1pred = []
    top5pred = []
    Instruction = "The goal is to complete the given sentence with one english word. Avoid punctuation or new line or space."
    for item in source:
    # item is e.g. 'The librarian documented which journalist the celebrities had '
        item = item.strip()
        prompt = Instruction + item 

        curl_req = 'curl https://api.openai.com/v1/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ' + key + '" \
        -d \'{"model": "text-davinci-002", "prompt": ' + '"' + prompt + '"' +', "temperature": 0.5, "n": 1, "max_tokens": 1, "logprobs": 5}\''
    
        # # Getting resposnse from gpt3
        gpt3result = subprocess.check_output(curl_req, shell=True)
        gpt3result = json.loads(gpt3result)
        # gpt3result = {'id': 'cmpl-5TKEoNR9pBCWrOkGhUOrep9bJU9TQ', 'object': 'text_completion', 'created': 1657670990, 'model': 'text-davinci-002', 'choices': [{'text': '\n', 'index': 0, 'logprobs': {'tokens': ['\n'], 'token_logprobs': [-0.00088851544], 'top_logprobs': [{'____': -8.674883, '\n': -0.00088851544, '________': -9.122221, '_____': -7.5989556, '________________': -10.313559}], 'text_offset': [165]}, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 34, 'completion_tokens': 1, 'total_tokens': 35}}
        # extracting the output text from gpt3 response
        top1pred_gpt3 = gpt3result['choices'][0]['logprobs']['tokens'][0]
        top5pred_gpt3 = gpt3result['choices'][0]['logprobs']['top_logprobs'][0]
        top5pred_gpt3 = list(top5pred_gpt3.keys())
        top1pred.append(top1pred_gpt3)
        top5pred.append(top5pred_gpt3)
    return top1pred, top5pred

def evaluation(modeldir, device, source, label, k, file_path):
    """
    modeldir [str] : model name
    device [str] : cpu or cuda
    source [list] : list of source sentences e.g. The librarian documented which journalist the celebrities had [MASK]'
    label [list] : list of gold labels
    k [int] : top k value minimum value 10 
    file_path [str] : file path where original source and label are present
    """
    # Define model and tokenizer
    file = open("result.txt", 'a')
    sensitivity_record = open("sensitivity.txt", 'a')
    # file.writelines([file_path, "\n", " | Model | top {} accuracy | top 10 accuracy | top 5 accuracy | top 1 accuracy".format(k), "\n"])
    
    print("Running experiment for ", modeldir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(modeldir)
    if modeldir.startswith("gpt2"):
        model = transformers.GPT2LMHeadModel.from_pretrained(modeldir).to(device)
    elif modeldir.startswith("t5"):
        model = transformers.T5ForConditionalGeneration.from_pretrained(modeldir).to(device)
    else: # for BERT, RoBERTa, ALBERT, DISTILBERT
        model = transformers.AutoModelForMaskedLM.from_pretrained(modeldir).to(device)
    model.eval()
    
    # Getting top k predictions from the model
    top_predictions = []
    x = 5 # index used to pick predictions, it only changes for roberta as its first predcition is always a space or ',' or '.'
    y = 0
    
    for item in source:
        # item is e.g. 'The librarian documented which journalist the celebrities had [MASK]'

        tokenized_input = tokenizer(item, return_tensors="pt")
        tokenized_input = tokenized_input.to(device)
        # print(tokenized_input)
        # print(tokenizer.decode(tokenized_input["input_ids"][0]))

        if modeldir.startswith('t5'):
            decoder_ids = tokenizer("<pad> <extra_id_0>", add_special_tokens=False, return_tensors="pt").input_ids
            decoder_ids = decoder_ids.to(device) 
            predictions = model(input_ids=tokenized_input.input_ids, decoder_input_ids=decoder_ids)
        else:
            predictions = model(**tokenized_input)

        if modeldir.startswith('roberta'):
            token = '<mask>'
            x = 6
            y = 1
        else:
            token ='[MASK]'

        if modeldir.startswith('gpt'):
            mask_index = -2  # -2 is position of last token
            y = 1
        elif modeldir.startswith('t5'): 
            mask_index = 1   # 1 is position of <extra_id_0>
        else: 
            for i, tok in enumerate(tokenized_input['input_ids'].reshape(-1)):
                if tok == tokenizer.convert_tokens_to_ids(token): 
                    mask_index = i       
                    
        predictions = predictions.logits

        softpred = torch.softmax(predictions[0, mask_index],0)
        top_inds = torch.argsort(softpred, descending=True)[:k].cpu().numpy()
        top_tok_preds = tokenizer.decode(top_inds, skip_special_tokens=True)
        top_predictions.append(top_tok_preds)
    
    # #of flips due to negation
    # neg-1500-simp, prediction-all (folder and dataset)
    flip = 0
    file_allsens = open('sensitivity-neg-all.txt', 'a')
    for i in range(0, len(top_predictions),2):
        list_top_pred = top_predictions[i].split(' ')[y] # e.g. ['fish', 'trout', 'species', 'mineral', 'protein']
        neg_list_top_pred  = top_predictions[i+1].split(' ')[y]
        print(list_top_pred, "---->", neg_list_top_pred)
        if list_top_pred != neg_list_top_pred:
            flip += 1
    file_allsens.writelines(["neg-1500-simp --->", modeldir , " | #flipped = ", str(flip), "\n"])
    
    # write all prediction store a file- includes affirmative and negation predictions
    file_allpred = open("predictions/predictions-all/neg-1500-simp/{}.txt".format(modeldir), 'w')
    for i in range(len(top_predictions)):
        list_top_pred = top_predictions[i].split(' ') # e.g. ['fish', 'trout', 'species', 'mineral', 'protein']
        file_allpred.writelines([str(list_top_pred),'\n'])
    print("prediction saved for ", modeldir)
    
    # # all sentence for role and only affirmative for neg-simp
    # step = 1
    # dataset = 'role-1500'
    # if 'neg' in file_path:
    #     step = 2 # for neg as we need only affirmative sentences
    #     dataset = 'neg-1500-simp'

    # # for role - 1500
    # # Accuracy for top 1, 5, 10 and 20 predictions
    # topkmatch = 0
    # top10match = 0
    # top5match = 0
    # top1match = 0
    # flipped = 0 # to keep track of how many times target word flips seeing 'not'
    # # file_pred = open("predictions/{}/{}.txt".format('dataset', modeldir), 'w')
    # file_pred = open("predictions/predictions-raw/neg-136-simp/{}.txt".format(modeldir), 'w')
    # # print(len(top_predictions))

    # for i in range(0, len(top_predictions), step):
    #     # print(i, source[i], label[i],top_predictions[i])

    #     list_top_pred = top_predictions[i].split(' ') # e.g. ['fish', 'trout', 'species', 'mineral', 'protein']
    #     file_pred.writelines([str(list_top_pred),'\n'])

    #     if label[i] in list_top_pred:
    #         topkmatch += 1
    #     if label[i] in list_top_pred[:10]:
    #         top10match += 1
    #     if label[i] in list_top_pred[:x]:
    #         top5match += 1
    #     if label[i] == list_top_pred[y]:
    #         top1match += 1
    #         # sensitivity for neg
    #         if 'neg' in file_path:
    #             if list_top_pred[y] != top_predictions[i+1].split(' ')[y]:
    #                 flipped += 1
    # # print(topkmatch)
    # topk_accuracy = step * topkmatch / len(top_predictions)
    # top10_accuracy = step * top10match / len(top_predictions)
    # top5_accuracy = step * top5match / len(top_predictions)
    # top1_accuracy = step * top1match / len(top_predictions)

    # print("model = ", modeldir)
    # print("Top 20 match = ", topk_accuracy)
    # print("Top 10 match = ", top10_accuracy)
    # print("Top 5 match = ", top5_accuracy)
    # print("Top 1 match = ", top1_accuracy)
    
    # file.writelines([file_path," | ", modeldir, " | ", str(topk_accuracy),  " | ", str(top10_accuracy),  " | ", str(top5_accuracy), " | ", str(top1_accuracy), '\n\n'])
    # if 'neg' in file_path:
    #     if top1match != 0:
    #         sensitivity_record.writelines([modeldir, " | ", "% target word changed = " , str(flipped/top1match), " | flipped = ", str(flipped)," | top 1 match = ", str(top1match),"\n"])
    # print("Completed experiment for ", modeldir)

def evaluation_gpt3(modeldir, key, source, label, file_path):
    """
    modeldir [str] : model name
    key [str] :open ai gpt3 key
    source [list] : list of source sentences e.g. The librarian documented which journalist the celebrities had [MASK]'
    label [list] : list of gold labels
    file_path [str] : file path where original source and label are present
    """
    top1matchgpt3 = 0
    top5matchgpt3 = 0
    flipped = 0 # for neg sensitivity
    step = 1
    if 'neg' in file_path:
        step = 2

    if 'role' in file_path:
        dataset = "role-1500"
    else:
        dataset = "neg-1500-simp"

    if modeldir.startswith("gpt3"): # for gpt3
        top1pred, top5pred = gpt3_responses(file_path, key) # get responses from gpt3 using curl request
        file = open("predictions/{}/gpt3_top1_prediction.txt".format(dataset), 'w')
        filetop5 = open("predictions/{}/gpt3_top5_prediction.txt".format(dataset), 'w')

        print(top1pred)
        print(top5pred)
        print(label)
        for i in range(0, len(label), step):
            top5pred[i] = [item.strip() for item in top5pred[i]]
            file.writelines([top1pred[i].replace("\n",''),"\n"])
            filetop5.writelines([str(top5pred[i]),"\n"])

            if label[i] == top1pred[i].strip():
                top1matchgpt3 +=1
                if 'neg' in file_path:
                    top1pred[i] != top1pred[i+1]
                    flipped += 1
            
            if label[i] in top5pred[i]:
                top5matchgpt3 += 1

    file = open("gpt3_result.txt", 'a')
    file.writelines([dataset," top1 accuracy = ", str(step*top1matchgpt3/len(label)), " | ", "top5 accuracy = ",  str(step*top5matchgpt3/len(label)), "\n"])

    print("top1 match", top1matchgpt3)
    if "neg" in file_path:
        if top1matchgpt3 != 0:
            file.writelines(["sensitivity is ",str(step*flipped/(top1matchgpt3)), "top1match=", top1matchgpt3])

def main():
    args = get_args()
    file_path = args.file_path
    modeldir = args.model_name_or_path
    key = args.key
    device = args.device
    source, label = process_data(file_path, modeldir)
    k = 20
   
    if modeldir.startswith("gpt3"):
        evaluation_gpt3(modeldir, key, source, label, file_path)
    else:
        evaluation(modeldir, device, source, label, k, file_path)
 
if __name__ == '__main__':
    main()