import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz


from data_utils import Dataset, Entity


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('popular')


confidence_level_per_length = {
    1: 100.0,
    2: 100.0,
    3: 100.0,
    4: 100.0,
    5: 80.0,
    6: 80.0,
    7: 80.0,
    8: 80.0,
    9: 75.0,
    10: 75.0
}


class NLPEngine:
    """
        based on regular expression
    """

    def __init__(self):
        self.total_entities = None
        self.stopwords = stopwords.words("english")

    def train(self, dataset):
        # collect all entities:
        self.total_entities = {}
        for example in dataset.examples:
            for entity in example.entities:
                if entity.ent_value not in self.total_entities.keys():
                    self.total_entities[entity.ent_value.lower().strip()] = entity
        self.total_entities = dict(sorted(self.total_entities.items()))
        return True

    def process(self, text):
        lower_text = text.lower()

        # tokenization
        # remove stop words
        tokens = [t for t in word_tokenize(lower_text) if t not in self.stopwords]

        result_entities = []
        seek_pos = 0
        for token in tokens:

            # seek the position of the token from the input query text
            start_pos = lower_text.find(token, seek_pos)
            end_pos = start_pos + len(token)

            # determine whether the token word is in dataset or not (eventhough there exists typo)
            max_similar = 0.0
            max_similar_entity = None
            for key, entity in self.total_entities.items():
                # ignore the word which has 2 or more different characters,
                # since there is limitation of typo as only 1 characters

                if not abs(len(key) - len(token)) <= 1:
                    continue

                _similar_ratio = fuzz.token_sort_ratio(token, key)
                if max_similar < _similar_ratio:
                    max_similar = _similar_ratio
                    max_similar_entity = entity

            if max_similar_entity is not None:

                threshold = confidence_level_per_length.get(len(max_similar_entity.ent_value),
                                                            100 - 100 / float(len(max_similar_entity.ent_value)))
                if max_similar >= threshold:
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
                    # self.process(example.text)
                    n_no_matched_gt += 1
                    print("\tfalse_negative:", gt_entity.ent_value)

            if len(pred_entities) > 0:
                n_no_matched_pred = len(pred_entities)
                # self.process(example.text)
                for pred_entity in pred_entities:
                    print("\tfalse_negative:", pred_entity.ent_value)

            true_positive += n_matched
            false_negative += n_no_matched_pred
            false_positive += n_no_matched_pred

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
    res = nlp.process("show me all flights from oenver to pittsurgh which serve a meal for the day after tomorrow")
    # res = nlp.process("book my a ticket from Boston to atlnta for this tursday")
    # print(res)

    # validate the solution ?
    nlp.evaluate_accuracy(dataset=_dataset)

    # true_positive: 10836
    # false_negative: 51
    # false_positive: 51
    # Total Score: 0.9907
