
# Gensim Keyword Extractor 

## Description

This app extracts keywords in a text document given the LDA topic model pretrained with
a given list of text files in a directory.

## Information on the available model
The current available model for keyword extraction is trained with 22 out of 24 NewsHour transcripts listed in
[batch2.txt](https://github.com/clamsproject/aapb-annotations/blob/9cbe41aa124da73a0158bfc0b4dbf8bafe6d460d/batches/batch2.txt).
Excluded files' names and reasons of exclusion are:
* `cpb-aacip-525-028pc2v94s`: File not found in the dataset
* `cpb-aacip_507-r785h7cp0z`: Contains no transcript but an error message

This model is trained with English stopwords removed. 
> TODO: some other default parameters of the current model 

## User instruction
### System requirements
* Requires Python3 with `clams-python`, `clams-utils`, `gensim`, `nltk`, and `scipy` to run the app locally.
* Requires an HTTP client utility (such as `curl`) to invoke and execute analysis.
* Requires docker to run the app in a Docker container 

Run `pip install -r requirements.txt` to install the requirements.

### Train a model with NewsHour transcripts using `lda.py`
> **NOTE:**
> If you only look to use the keyword extractor app instead of training your own model, 
> please skip this section and follow instructions in the next section. 

After getting into the working directory, run the following line on the target dataset:

`python lda.py --dataPath path/to/target/dataset/directory`

By running this line, `lda.py` does 2 things:
* cleans all transcripts in a given directory.
* generates the pretrained LDA model that stores the dictionary and the corpus. 
Currently, this file is not allowed to be renamed, or it affects running `cli.py` later on.

> TODO: some other parameters of lda.py 

### Extract keywords using the app 

General user instructions for CLAMS apps are available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

To run this app in CLI:

`python cli.py --optional_params <input_mmif_file_path> <output_mmif_file_path>`

2 types of input `MMIF` files are acceptable here:
* The ones that are generated through `clams source text:/path/to/the/target/txt/file` to extract keywords for a single
text document.
* The ones whose last view containing TextDocument(s) is the view to extract keywords from.

Default number of keywords extracted from a given text document is 10. If the number of extracted keywords is required 
to be different from 10, when running `cli.py`, add `--topN` and a corresponding integer value. 

Two scenarios may be seen if the input text document is too short:
1. If the number of tokens in a text document is smaller than the value of `topN`, 
then no keywords will be extracted. 
2. If the text contains lots of stopwords, then the number of extracted keywords can be less than the value of `topN`,
because the app ignores all stopwords when finding keywords. 

### Configurable runtime parameter

For the full list of parameters, please refer to the app metadata from the [CLAMS App Directory](https://apps.clams.ai) 
or the [`metadata.py`](metadata.py) file in this repository.

