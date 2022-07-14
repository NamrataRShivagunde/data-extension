

file = open("neg-1500-simp-generated.txt", 'a')
with open("data/intermediate-data/neg-1500-simp-generated.txt") as f:
    text = f.readlines()
    for line in text:
            sent = line.split(" ")
            lastword = sent[-1]
            vowel = 'aeiou'
            if lastword[0] in vowel:
                line = line.replace("(a/an)","an")
            else:
                line = line.replace("(a/an)","a")
            print(line)
            file.writelines(line)
            
            
