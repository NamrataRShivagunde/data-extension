from numpy import mod
import transformers
import torch
import argparse
import re

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
    """Convert the examples into source and label

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
    
    # Read line by line, remove (, ), , and replace last word with [MASK]
    # Append last word to label file
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
    # print(source, label)
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
    k = 10
    for item in source:
        # tokenized_text = tokenizer.tokenize(item)
        # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        # tokens_tensor = torch.tensor([indexed_tokens])

        tokenized_input = tokenizer(item, return_tensors="pt").input_ids
        tokenized_input = tokenized_input.to(device)

        if modeldir.startswith('t5'):
            decoder_ids = tokenizer("<pad> <extra_id_0>", add_special_tokens=False, return_tensors="pt").input_ids
            decoder_ids = decoder_ids.to(device) 
            predictions = model(input_ids=tokenized_input, decoder_input_ids=decoder_ids)
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
            # mask_index = ((tokenized_text["input_ids"][0] == tokenizer.convert_tokens_to_ids(token)).nonzero(as_tuple=True)[0]).item()
        
        predictions = predictions.logits
        softpred = torch.softmax(predictions[0, mask_index],0)
        top_inds = torch.argsort(softpred, descending=True)[:k].cpu().numpy()
        top_tok_preds = tokenizer.decode(top_inds, skip_special_tokens=True)
        top_predictions.append(top_tok_preds)

    # textfile = open("predictions/{}.txt".format(modeldir), "w")
    # for item in top_predictions:
    #     textfile.write(item + "\n")      
    # textfile.close()

    # Accuracy
    topkmatch = 0
    top1match = 0
    for i in range(len(top_predictions)):
        # print(label[i], "--->", top_predictions[i])
        if label[i] in top_predictions[i]:
            topkmatch += 1
        if label[i] == top_predictions[i][0]:
            top1match += 1

    topk_accuracy = topkmatch/len(top_predictions)
    top1_accuracy = top1match/len(top_predictions)

    print("Top 5 match = ", topk_accuracy)
    print("Top 1 match = ", top1_accuracy)
    
    
    file = open("result.txt", 'a')
    file.writelines([file_path, "----" ,modeldir, " ----Top 5 match accuracy = ", str(topk_accuracy), "----Top 1 match accuracy = ", str(top1_accuracy) ,'\n'])
    print("Completed experiment for ", modeldir)
if __name__ == '__main__':
    main()