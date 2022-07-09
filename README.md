# data-extension

NEG-SIMP is an extension of NEG-SIMP from (ettinger, 2020) using 56 categories and their subcategories from original paper (Battig, 1969). 
To extend the NEG-SIMP dataset run 
        python generate-data.py --dataset 'neg' --key [open ai key]

to evaluate models on the extended NEG-SIMP dataet run
        python evaluation.py 'data/neg-simp.txt' roberta-base
        python evaluation.py 'data/role-88-generated.txt' rsoberta-base

to evaluate all models run
        python run-all-models.py 'data/role-88-generated.txt'


References-
Battig, William F. and William Edward Montague. “Category norms of verbal items in 56 categories A replication and extension of the Connecticut category norms.” Journal of Experimental Psychology 80 (1969): 1-46. 

Ettinger, A. (2020). What BERT is not: Lessons from a new suite of psycholinguistic diagnostics for language models. Transactions of the Association for Computational Linguistics, 8, 34-48. https://arxiv.org/abs/1907.13528 