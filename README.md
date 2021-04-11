# Extracting entities with typos

### Task:
- Design an algorithm to extract entities from expressions.
- Be robust to typos. Note that the typos are limited to a single character.
- Approach this problem as a real NLP problem and adhere to best practices.
There is no guarantee on the typos in future expressions that will be sent to the chatbot, except for the typo limit of one.
- The code should be well-structured, documented, and contain coding best practices.
- It is not mandatory to design everything from scratch. There is a possibility to use an external library.
- You can assume that all cities and days mentioned in the dataset are the only possible values.

### setup

Under the assume that there exists the python3 (or python virtual env) already
```pip3 install -r requirements.txt```

download the dependencies for spacy
``` pip3 install -U pip setuptools wheel
    python3 -m spacy download en_core_web_sm (linux/mac)
    # (in the case of window virtual env) venv\Scripts\python.exe -m spacy download en_core_web_sm (windows under the assume that venv:virtual env is ready)
```

download the dependencies for nltk 
```nltk.data.find('tokenizers/punkt')```
