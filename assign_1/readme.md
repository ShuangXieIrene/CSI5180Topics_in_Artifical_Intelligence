
# Text classification

## Report

## Compatibility
* Python 2.7 / Python 3.4
* Linux / OSX / Windows

## Data
[Data](https://drive.google.com/open?id=1WWOcDd50uqInZF9tEwMUtzdT6MX7Vr2i)

## Environment
The code is runned in a 2.5 GHz Intel Core i7 with MacOS systems.

## Usage
- Please find the assignment report in the file report_assign1.pdf.
- The code for pre-processing documents is in the './source_code/preprocess_data/' folder.
- You can run the sk_classify_10fold.py code to finish classify the documents in the folder './source_code/'. Besides, in the folder './source_code/result/', the evaluation metrics are saved in 'log.txt' file and the classifier models after training are in pkl format.
- Another version without using sklearn is implemented in Naive Bayes folder, './dataset.py' is a class used to pre-process data, and 'bayesclassifier.py' is a class used to implement Naive Bayes classifier algorithm. However, we failed to get the feature distribution due to the sparse matrix (the mean and var cannot be computed when there are lots of zero in sparse matrix).

## Model
[Classifier Models](https://drive.google.com/open?id=1HLuvacNi5W6AyfkLKtDVOgL5r4g8Ykxq)




