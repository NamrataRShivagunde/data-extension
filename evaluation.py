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
                line_source = line.replace(line_label, '<mask>')
            elif modeldir.startswith("gpt"):
                line_source = line.replace(line_label, '')
            elif modeldir.startswith('t5'):
                line_source = line.replace(line_label, '<extra_id_0>')
            else:
                line_source = line.replace(line_label, '[MASK].')
            label.append(line_label.rstrip("\n"))
            source.append(line_source)
    return source, label
   

def main():
    args = get_args()
    file_path = args.file_path
    modeldir = args.model_name_or_path
    device = args.device
    source, label = process_data(file_path, modeldir)
    k = 20
    file = open("result.txt", 'a')
    sensitivity_record = open("sensitivity.txt", 'a')
    # file.writelines([file_path, "\n", " | Model | top {} accuracy | top 10 accuracy | top 5 accuracy | top 1 accuracy".format(k), "\n"])

    # Define model and tokenizer
    print("Running experiment for ", modeldir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(modeldir)
    if modeldir.startswith("gpt"):
        model = transformers.GPT2LMHeadModel.from_pretrained(modeldir).to(device)
    elif modeldir.startswith("t5"):
        model = transformers.T5ForConditionalGeneration.from_pretrained(modeldir).to(device)
    elif modeldir.startswith("albert"):
        model = transformers.AlbertForMaskedLM.from_pretrained(modeldir).to(device)
    elif modeldir.startswith("distilbert"):
        model = transformers.DistilBertForMaskedLM.from_pretrained(modeldir).to(device)
    else: # for BERT, RoBERTa
        model = transformers.AutoModelForCausalLM.from_pretrained(modeldir).to(device)
    model.eval()
    
    # Getting top k predictions from the model
    top_predictions = []
    
    for item in source:
        # item is e.g. 'The librarian documented which journalist the celebrities had [MASK]'

        tokenized_input = tokenizer(item, return_tensors="pt")
        tokenized_input = tokenized_input.to(device)

        if modeldir.startswith('t5'):
            decoder_ids = tokenizer("<pad> <extra_id_0>", add_special_tokens=False, return_tensors="pt").input_ids
            decoder_ids = decoder_ids.to(device) 
            predictions = model(input_ids=tokenized_input.input_ids, decoder_input_ids=decoder_ids)
        else:
            predictions = model(**tokenized_input)

        if modeldir.startswith('roberta'):
            token = '<mask>'
        else:
            token ='[MASK]'

        if modeldir.startswith('gpt'):
            mask_index = -2  # -2 is position of last token
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

    step = 1
    if 'neg' in file_path:
        step = 2 # for neg as we need only affirmative sentences

   
    # for role - 1500
    # Accuracy for top 1, 5, 10 and 20 predictions
    topkmatch = 0
    top10match = 0
    top5match = 0
    top1match = 0
    flipped = 0 # to keep track of how many times target word flips seeing 'not'

    print("----------------",top_predictions)
    for i in range(0, len(top_predictions), step):
        list_top_pred = top_predictions[i].split(' ')

        # sensitivity for neg
        if 'neg' in file_path:
            if list_top_pred[0] != top_predictions[i+1].split(' ')[0]:
                flipped += 1
                print(flipped)
    
        if label[i] in list_top_pred:
            topkmatch += 1
        if label[i] in list_top_pred[:10]:
            top10match += 1
        if label[i] in list_top_pred[:5]:
            top5match += 1
        if label[i] == list_top_pred[0]:
            top1match += 1


    topk_accuracy = step * topkmatch / len(top_predictions)
    top10_accuracy = step * top10match / len(top_predictions)
    top5_accuracy = step * top5match / len(top_predictions)
    top1_accuracy = step * top1match / len(top_predictions)

    print("model = ", modeldir)
    print("Top 20 match = ", topk_accuracy)
    print("Top 10 match = ", top10_accuracy)
    print("Top 5 match = ", top5_accuracy)
    print("Top 1 match = ", top1_accuracy)
    
    print(flipped)
    file.writelines([file_path," | ", modeldir, " | ", str(topk_accuracy),  " | ", str(top10_accuracy),  " | ", str(top5_accuracy), " | ", str(top1_accuracy), '\n\n'])
    if 'neg' in file_path:
        sensitivity_record.writelines([modeldir, " | ", "% target word changed due to negation = " , str(2 * flipped/len(top_predictions)), "\n"])
    print("Completed experiment for ", modeldir)


if __name__ == '__main__':
    main()