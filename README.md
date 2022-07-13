# data-extension

NEG-SIMP is an extension of NEG-SIMP from (ettinger, 2020) using 56 categories and their subcategories from original paper (Battig, 1969). 
To extend the NEG-136-SIMP dataset run 
        python generate-data.py --dataset 'neg' 

To extend the ROLE-88 dataset run
        python generate-data.py --dataset 'role' --key [open ai key]

to evaluate a particular model on the extended NEG-1500-SIMP dataset, run
        python evaluation.py [FILENAME] [MODELNAME]
        e.g.
        python evaluation.py 'data/neg-simp-generated.txt' roberta-base
        python evaluation.py 'data/role-88-generated.txt' roberta-base

to evaluate all models run
        python run-all-models.py 'data/neg-1500-simp-generated.txt'
        python run-all-models.py 'data/role-1500-generated.txt'

result.txt has the result 
        its top k prediction accuracy for top 20 | top 10 | top 5 | top 1

sensitivity.txt has the neg sensitivity result

all predictions top 20 are saved in prediction folder.

References-
Battig, William F. and William Edward Montague. “Category norms of verbal items in 56 categories A replication and extension of the Connecticut category norms.” Journal of Experimental Psychology 80 (1969): 1-46. 

Ettinger, A. (2020). What BERT is not: Lessons from a new suite of psycholinguistic diagnostics for language models. Transactions of the Association for Computational Linguistics, 8, 34-48. https://arxiv.org/abs/1907.13528 