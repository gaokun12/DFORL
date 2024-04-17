# README 

DFOL is a software which learns first-order logic progrrmas from larger knowledge graph.

## Experimantal Environments:

The list shows the explicit software in the DFOL and the corresponding version. The versions in the parameter indicate the actual software versions used in the author's environment.

- Python 3.6.13 and above (3.10.4)

- Tensorflow  2.6 and above (2.8.0)

  ```shell
  pip install tensorflow==2.8
  ```

- pyDatalog  (0.17.1)
  
  Note: When installing the pyDatalog under Python 3.10, we need to download the source code manually and change 'from  collections import *' to 'form collections.abc import *'. Then, install pyDatalog through the 'pip install' command. 

  ```shell
  pip install pyDatalog
  ```

- Numpy (1.22.3)

  ```
  pip install numpy==1.22.3
  ```

- pickleshare  (0.7.5)

  ```
  pip install pickleshare
  ```

- seaborn (0.11.2)

  ```
  pip install seaborn
  ```

- matplotlib (3.5.1)

  ```
  pip install matplotlib
  ```

- pydot (1.4.2)

  ```shell
  pip install pydot
  ```

- graphviz (0.19.1)

  ```shell
  pip install graphviz	
  ```

## Learning small ILP datasets:

Entering the code directory.

The means of arguments:

- g: Generating the trainable dataset from relational facts using the proposed propositionalization method;

- d: The name of the dataset. In the code, the name of the directory where the task is located in. 

- p: The name of target predicate, which is the same as the name of the file *.nl* under the ‘dataset_name/data/’ folder. 

- cur: The flag controlling whether beginning the training process with the curriculum learning strategy. 

- ft: The soundness filter with the default value 0.3.

- vd: The variable depth with the default value 1. 

- lt: The curriculum learning times

- amg: Learning with probabilistic datasets, the standard rate is [3] in default.

- mis: Learning with mislabeled datasets. 

- checkpl: Check the accuracy for a logic program head by a relation.

- ap: Generate logic programs head by all predicates in the task, and generate HINT@n and MRR indicators. 

- cap: After obtaining logic programs head bt all predicates in the task, running this to get mean accuracy of all logic programs. 

- lar: Whether the dataset is a large knowledge base dataset. Some known small dataset includes (1) All classical inductive logic programming datasets; (2) Nations dataset, Countries datasets, and UMLS datasets. If the large flag is open, then the program will use the proposed propositionalization method to save the data into TFRecords format. (0 as the default.)

- bs: The batch_size when training the model. The default value of the batch_size is 64. (If the result is not accurate enough, please increase the batch size.)

<!-- - ete: This flag indicates the program will use the data predicted from the embedding-based link prediction model to generate trainable data. At the same time, the trainable data are stored in the TFRecords format. When this flag is open, the large flag is open in the default. Hence, we do not need to open 'lar' flag mannually. (0 as the default.) -->

- percent: When learning the large datasets, we only do the sampling propositionalization. The percent means the ratio of considered substitutions of all substitutions. This flag only makes roles during the data generating phase. The default value is 1, which means we generate data according to the substitutions computed based on all objects.  

- ver: The details preview information when training a model. [1]: The model logs the information at each step/iteration. [2]: The model logs the information at each batch. (1 as the default.) 

- walk_n: We use the walk_n flag to indicate whether the sampling algorithm is running. The default value is 0, which means the sampling algorithm is deactivated. If the value of walk_n is not 0 but floats in (0,1), then it means the sampling algorithm is open and the value indicates the ratio of considered target positive examples to all target positive examples. 

- walk_c: We use the walk_c flag to determine DFOL to check the accuracy of the covered test positive examples on which file.  This flag only reflects the accuracy computation process after finishing the training process at the current time. The default value is 0, which means we check the accuracy from '.nl' file in default. If the value id 1, which means DFOL check accuracy on '.onl' file. After using the sampling algorithm, the '.nl' consists of the **target** positive examples and associated examples. But '.onl' file consists of all positive examples and background knowledge. 


Take the even as an example. 

```shell
python DFOL/model/main.py -g 1 -d even -p even -cur 1 -ft 1
```

Note: If the final accuracy is not satisfied (not 1), please use the command again because DFOL can use the prior generated results as prior knowledge to reduce the search space for the following learning process. 

## Learning ambiguous datasets:

Taking the mislabeled lessthan as an example:

```shell
python DFOL/model/main.py  -d lessthan_mis -p lessthan  -ft 1 -mis 1
```

Taking the probabilistic lessthan as an example:

```shell
python DFOL/model/main.py -d lessthan_pb -p lessthan  -ft 1 -amg 1
```

## Learning knowledge base:

- Learning Countries dataset:

  - Taking the S1 sub dataset as an example, and setting the soundness filter as 0.5 in this example.

    ```shell
    python DFOL/model/main.py -d locatedIn_S1 -p locatedIn  -ft 0.5 -g 1 -cur 1
    ```
  - Taking the S3 subdataset as an example, and setting the soundness filter as 0.3 in this example. In this task, we need at least 4 different variable to describe the hypothesis, then the number of all substitutations is quite large. We use 'percent' and 'lar' flag in this task. In addition, we set the percent as 1%, which measn we consider only 1% substitutations in the propositionalization process.  

    ```shell
    python DFOL/model/main.py -d locatedIn_S3 -p locatedIn  -ft 0.3 -g 1 -vd 2 -lar 1 -percent 0.01 -cur 1
    ```


- Learning UMLS and Nations dataset:

  - Taking UMLS as an example, and taking the soundness filter as 0.3:

    0. Sample the data when the dataset is too large to perform the propositionalization method directly. 

        ```shell
        python DFOL/model/main.py -d wn18rr_sub -p _hypernym -walk_n 3
    1. Generating only a logic program head by a predicate in the task, i. e., head by *isa* predicate. (Note: User should duplicate the *task_name*.nl file and rename it to *target_relation*.nl)

       ```shell
       python DFOL/model/main.py -d umls -p isa -ft 0.3 -g 1 -cur 1
       ```

    2. Generating all logic programs head by the predicate shown in the task. 

       ```shell
       python DFOL/model/main.py -d umls -p umls  -ft 0.3  -ap 1 -lar 0 
       ```
       (dataset is generated only in memory)

        or
        ```shell
        python DFOL/model/main.py -d kinship -p kinship  -ft 0.3  -ap 1 -lar 1 
        ``` 
        (dataset is generated associated with disk)

        or 
        ```shell
        py DFOL/model/main.py -d fb15k-237-sub -p fb15k-237-sub -ft 0.3 -lar 1 -ap 1  -walk_n 1 (slow speed)
        ```
        (set the sampling method with 100% positive examples in advance, the algorithm will reduce the size of seen positive examples if all positive examples exceed the limit of memory)

        or 
        ```shell
        py DFOL/model/main.py -d wn18rr_sub -p bottom_up -ft 0.3 -bap 1  -walk_n 0.01 -walk_c 0 
        ```
        (set the bottom-up manner to generate the logic programs. The sampling rate is 0.01, the focus model is open and lar is close)

    3. After generating all logic programs, compute the MRR and HITS@(1,3,10) on the testing data:

       ```shell
       python DFOL/model/main.py -d umls  -ind 1 
       ```
       or 
       ```shell
       python DFOL/model/main.py -d wn18rr_sub  -ind 1 -walk_c 1 
       ```
       (if we use bottom up strategy to generate logic programs, and we want to check MRR and HIT, please run *check soundness* function to update the soundness of each rule in all facts. )

    4. After generating all logic programs, compute the accuracy of the testing data:

        (According to the current check policy, i.e., when a tested fact is satisfied iff the test fact can be inferred by any probabilistic rule generated in the best.pl file, the HITS@10 value is equal with the accuracy in UMLS and Nations dataset.)
        ```shell
        python DFOL/model/main.py -d umls  -cap 1 
        ```
        or 
       ```shell
       python DFOL/model/main.py -d wn18rr_sub   -cap 1 -walk_c 1 -tfn (best/neurlp)
       ```

    5.1 Compute the accuracy for an explicit logic program, i.e., head by *isa* predicate:

       ```shell
       python DFOL/model/main.py -d umls -p isa -checkpl 1 -checktrain 0
       ```
    
    5.2 Compute the soundness for an explicit logic program, i.e., head by *isa* predicate:

       ```shell
       python DFOL/model/main.py -d umls -p isa -checkpl 1 -checktrain 1
       ```

    6. Focus model: to learn the logic program with a predicate on a larger dataset.
        ```shell
        py NLP/model/main.py -d wn18rr_sub -p _has_part -walk_n 0.01 -walk_c 0 -focus 1 -lar 0
        ```
        (The walk_n should be small enough to handle the data when -lar flag is closed)
    
    7. Check the average Soundness of all Logic programs: to learn the logic program with a predicate on a larger dataset. (remove #TEST tags in .onl database)
    ```shell
    py NLP/model/main.py -d wn18rr  -walk_c 1 -sodche 1 -checktrain 1 (remove '#TEST' tags in '.onl' database in advance)
    ```
