#How to use
1. Place the training data in data/Challenge, and test data in data/Evaluation
2. Run src/IO.py, and get csv files that will be used for the next steps. 
3. Run src/model-size.py to get the sizes of models with various parameters.
4. Run src/table-maker.py, and get the estimated performance of each model under various parameters.
5. Select the best pair of parameter by using the tables from 3 and 4.
6. Run src/method.py to get the final model.


#Scripts

##IO.py
Read training data in data/Challenge and test data in data/Evaluation, and generate the csv files for SNPeff.
To be more precise, this script outputs three csv files:
1. out/SNPeff/SNPeff\_train.csv
2. out/SNPeff/SNPeff\_test.csv
3. out/SNPeff/variant\_gene\_list.csv

###Sample usage
```bash
python3 src/IO.py
```


##method.py
For given training data in data/Challenge, test data in data/Evaluation, csv files in out/SNPeff, and (d\_cn, k\_var) from argv, 
output the shallow network models trained by our method.

###Sample usage
```bash
python3 src/method.py --cn 0.1 --snpeff 260 --repeat 10 --path out/model/
```


##table-maker.py
For given training data in data/Challenge, csv files in out/SNPeff, and n and a range of (d\_cn, k\_var) from argv,
output a table that describes the performance of our method using n-fold cross validation under each (d\_vn, k\_var).
More precisely, this script outputs csv files:
1. [path]/exact\_aucs.csv
2. [path]/exact\_accs.csv
3. [path]/approx\_aucs.csv
4. [path]/approx\_accs.csv

###Sample usage
```bash
python3 src/table-maker.py --cn-start 0 --cn-step 0.01 --cn-stop 0.4 --snpeff-start 0 --snpeff-step 10 --snpeff-stop 600 --repeat 10 --path out/table/
```


##model-size.py
For given training data in data/Challenge, csv files in out/SNPeff, and a range of (d\_cn, k\_var) from argv, 
output the table of the size of models under each (d\_cn, k\_var).

###Sample usage
```bash
python3 src/model-size.py --cn-start 0 --cn-step 0.01 --cn-stop 0.4 --snpeff-start 0 --snpeff-step 10 --snpeff-stop 600 --path out/modelTMP/num_cands.csv
```


