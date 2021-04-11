import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


from sentence_transformers import SentenceTransformer, util
from data_utils import Dataset, Entity


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('popular')


class NLPEngine:
    """
        based on DeepLearning NLP model BERT
    """

    def __init__(self):
        self.total_entities = {}

        self.stopwords = stopwords.words("english")

        # self.model = SentenceTransformer('paraphrase-distilroberta-base-v1')  # 0.4278
        self.model = SentenceTransformer('stsb-roberta-base')  # 0.5110
        # self.model = SentenceTransformer('msmarco-distilbert-base-v3')  # 0.2825
        # self.model = SentenceTransformer('nq-distilbert-base-v1')  # 0.4747

    def train(self, dataset):
        # collect all entities:
        for example in dataset.examples:
            for entity in example.entities:
                if entity.ent_value not in self.total_entities.keys():
                    self.total_entities[entity.ent_value.lower().strip()] = entity
        return True

    def process(self, text):
        lower_text = text.lower()

        # tokenization
        tokens = word_tokenize(lower_text)

        result_entities = []
        seek_pos = 0
        for token in tokens:

            # remove the english stop words
            if token in self.stopwords:
                continue

            # seek the position of the token from the input query text
            start_pos = lower_text.find(token, seek_pos)
            end_pos = start_pos + len(token)

            # determine whether the token word is in dataset or not (even though there exists typo)
            max_similar = 0.0
            max_similar_entity = None
            for key, entity in self.total_entities.items():

                # ignore the word which has 2 or more different characters,
                # since there is limitation of typo as only 1 characters
                if not abs(len(key) - len(token)) <= 1:
                    continue

                _similar_score = util.pytorch_cos_sim(self.model.encode(entity.ent_value), self.model.encode(token))
                if max_similar < _similar_score:
                    max_similar = _similar_score
                    max_similar_entity = entity

            if max_similar_entity is not None and max_similar >= 1.0 - 1.0 / (len(max_similar_entity.ent_value) - 1):
                result_entities.append(Entity(ent_type=max_similar_entity.ent_type,
                                              ent_value=max_similar_entity.ent_value,
                                              start_index=start_pos,
                                              end_index=end_pos))
                seek_pos = end_pos

        return result_entities  # return list of entities found in text (type Entity)

    def evaluate_accuracy(self, dataset):
        """
        TP: TruePositive - Object is there and the model detects it.
        FN: FalseNegative - Ground truth is present but the model failed to detect the object.
        FP: FalsePositive - Ground truth is present but the model detects the wrong object.
        TN: TrueNegative - nothing was not detected - No meaning - Ignore
        """
        true_positive = 0
        false_negative = 0
        false_positive = 0
        # true_negative = 0  (entity detection task, the false_positive has not meaning, so ignore this)

        for example in dataset.examples:
            pred_entities = self.process(text=example.text)

            gt_entities = example.entities

            n_matched, n_no_matched_gt, n_no_matched_pred = 0, 0, 0
            for gt_entity in gt_entities:
                flag_matched = False
                for pred_entity in pred_entities[:]:
                    if gt_entity.ent_value == pred_entity.ent_value and \
                            gt_entity.start_index == pred_entity.start_index and\
                            gt_entity.end_index == pred_entity.end_index:
                        true_positive += 1
                        pred_entities.remove(pred_entity)

                        flag_matched = True
                        n_matched += 1

                if flag_matched is False:
                    n_no_matched_gt += 1
                    print("false_negative:", example.text)
                    print(f"\t{gt_entity.ent_value}")

            if len(pred_entities) > 0:
                n_no_matched_pred = len(pred_entities)
                print("false_negative:", example.text)
                for pred_entity in pred_entities:
                    print(f"\t{pred_entity.ent_value}")

            true_positive += n_matched
            false_negative += n_no_matched_pred
            false_positive += n_no_matched_pred
            print(true_positive, false_negative, false_positive)

        acc = true_positive / (true_positive + false_positive + false_negative)

        print(f"true_positive : {true_positive}")
        print(f"false_negative: {false_negative}")
        print(f"false_positive: {false_positive}")

        print(f"Total Score: {acc:.4}")


if __name__ == "__main__":
    # create NLPEngine
    nlp = NLPEngine()

    # read dataset
    _dataset = Dataset.read_dataset("dataset.json")

    # train model
    nlp.train(_dataset)

    # test expression
    res = nlp.process("book my a ticket from Boston to atlnta for this tursday")
    for ent in res:
        print(ent.ent_value, ent.start_index, ent.end_index)

    # validate the solution ?
    nlp.evaluate_accuracy(dataset=_dataset)
