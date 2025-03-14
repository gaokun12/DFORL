# README ☺️

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
- bs: The batch_size when training the model. The default value of the batch_size is 32. (If the result is not accurate enough, please increase the batch size. 32 as the default.)
<!-- - ete: This flag indicates the program will use the data predicted from the embedding-based link prediction model to generate trainable data. At the same time, the trainable data are stored in the TFRecords format. When this flag is open, the large flag is open in the default. Hence, we do not need to open 'lar' flag mannually. (0 as the default.) -->
- percent: When learning the large datasets, we only doing the subsamping propositionalization. The percent means the ratio of considers substitutations of all substitutations. This flag only makes roles during the data generating phase. 
- ver: The deatils preview information when training a model. [1]: The model logs the information at each step/iteration. [2]: The model logs the information at each batch. (1 as the default.) 
- walk: We use walk flag to indicate whether the walk algorithm is open when the data is large. This flag only reflects the accuracy computation process after finishing the traing process in the current time.


Taking the even as an example. 

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

    1. Generating only a logic program head by a predicate in the task, i. e., head by *isa* predicate. (Note: User should should duplicate the *task_name*.nl file and rename it to *target_relation*.nl)

       ```shell
       python DFOL/model/main.py -d umls -p isa -ft 0.3 -g 1 -cur 1
       ```

    2. Genretaing all logic programs head by the predicate shown in the task. And compute MRR and HITS@n:

       ```shell
       python DFOL/model/main.py -d umls -p umls  -ft 0.3  -ap 1
       ```

    3. After generating all logic programs, compute the accuracy on the testing data:

       ```shell
       python DFOL/model/main.py -d umls  -cap 1 
       ```

    4. Compute the accuracy for an explicit logic program, i. e., head by *isa* predicate:

       ```shell
       python DFOL/model/main.py -d umls -p isa -checkpl 1 
       ```
