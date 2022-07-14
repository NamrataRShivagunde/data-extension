# python sensitivity-test.py
# After the predictions are stored in prediction folder
file = open("neg-1500-simp-lastwords", 'w')
file_pred = open("neg-1500-simp-lastwords_pred", 'w')
file_sens = open("neg-1500-simp-sensitivity", 'w')

# fixed as it is based on true labels
lastwords = []
noun = []
# store true completion
# with open("data/neg-1500-simp-generated.txt") as f:
with open("data/neg-136-simp.txt") as f:
    text = f.readlines()
    for line in text:
            sent = line.split(" ")
            lastword = sent[-1]
            lastword = lastword.split(',')[0]
            nounword = sent[1]
            lastwords.append(lastword)
            noun.append(nounword)
            file.writelines([lastword + '\n'])

# lastword changes as its based on model prediction
# compare the negated version of sentence and check if it is completed by thge true completion of affirmative sentences.
models = ['gpt2','t5-small','albert-base-v1','roberta-base', 'bert-base-uncased']

for model in models:
    print(model)
    lastwords_pred = []
    with open("predictions/predictions-all/neg-1500-simp/{}.txt".format(model)) as pred:
        predictions = pred.readlines()
        for line in predictions:
            line = line.strip('[')
            line = line.strip(']')
            line = line.split(" ")
            item = 0
            if model.startswith('roberta'):
                item = 1
                print('check')
            line = line[item].strip(',')
            line = line[1:-1]
            
            lastwords_pred.append(line)
            file_pred.writelines([line + '\n'])
        #compare true completion of affirmative sentence to prediction of negation. if matches +1. these many negated senetnces have wrong completion.
        count = 0
        # print(len(lastwords))
        # print(len(lastwords_pred))
        for i in range(0, len(lastwords)-1, 2):
            print(i)
            print(model, "  ", str(noun[i+1]) ,"  lastword----->",str(lastwords[i]),"  pred word --->  ", str(lastwords_pred[i+1]))
            if (lastwords[i] == lastwords_pred[i+1]) or (noun[i+1] == lastwords_pred[i+1]):
                print("-----------------")
                count += 1
        print(count)

        file_sens.writelines([model, " model assigns wrong completion to (out of 36) ---> ", str(count), "\n"])

    