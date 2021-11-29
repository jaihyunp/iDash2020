##IO
Read training data in data/Challenge and test data in data/Evaluation, and generate the csv files for SNPeff.
More precisely, it outputs three csv files:
1. out/SNPeff/SNPeff\_train.csv
2. out/SNPeff/SNPeff\_test.csv
3. out/SNPeff/variant\_gene\_list.csv

###Sample usage
```bash
python3 src/IO.py
```


##Model
Read training data in data/Challenge, test data in data/Evaluation, and csv files in out/SNPeff, and given parameters of CN, SNPeff and the number of repeats,
output the shallow network models trained by our method.

###Sample usage
```bash
python3 src/method.py --cn 0.1 --snpeff 260 --repeat 10 --path out/model/v21.11.19.tmp/ > out/model/v21.11.19.tmp/terminal.txt
```


##Table
Read training data in data/Challenge, test data in data/Evaluation, and csv files in ./out/SNPeff, and given parameters of CN, SNPeff and the number of repeats,
output the table that indicates the performance of our method under various parameters. 
More precisely, it outputs csv files:
1. [path]/exact\_aucs.csv
2. [path]/exact\_accs.csv
3. [path]/approx\_aucs.csv
4. [path]/approx\_accs.csv
5. [path]/num\_cands.csv
6. [path]/ctrs.csv

###Sample usage
```bash
python3 src/table-maker.py --cn-start 0 --cn-step 0.01 --cn-stop 0.4 --snpeff-start 0 --snpeff-step 10 --snpeff-stop 600 --repeat 10 --path out/table/v21.11.19tmp/
```
