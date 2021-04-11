import random
import nltk
import spacy
from nltk.corpus import stopwords
from spacy.training.example import Example


from data_utils import Dataset, Entity


nlp = spacy.blank("en")
# nlp = spacy.load('en_core_web_sm')
ner = nlp.create_pipe("ner")
nlp.add_pipe("ner")


# pip3 install -U pip setuptools wheel
# pip3 install -U spacy
# python3 -m spacy download en_core_web_sm (linux/mac)
# venv\Scripts\python.exe -m spacy download en_core_web_sm (windows under the assume that venv:virtual env is ready)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('popular')


class NLPEngine:
    """
        based on DeepLearning NLP model BERT
    """

    def __init__(self, nlp=None):
        self.total_entities = {}

        self.stopwords = stopwords.words("english")
        self.custom_nlp = nlp

    def train(self, dataset):
        # collect all entities:
        for example in dataset.examples:
            for entity in example.entities:
                if entity.ent_value not in self.total_entities.keys():
                    self.total_entities[entity.ent_value.lower().strip()] = entity

        # prepare the standardized spacy entry:
        train_data = []
        for example in dataset.examples:
            entities = []
            for e in example.entities:
                entities.append((e.start_index, e.end_index, e.ent_type))
            spacy_entry = (example.text, {"entities": entities})
            train_data.append(spacy_entry)
        # Adding labels to the `ner`
        for _, annotations in train_data:
            for ent in annotations.get("entities"):
                ner.add_label(ent[2])

        # Start the training
        nlp.begin_training()
        # Loop for 40 iterations
        for itn in range(10):
            # Shuffle the training data
            random.shuffle(train_data)
            losses = {}
            # Batch the examples and iterate over them
            for batch in spacy.util.minibatch(train_data, size=4):
                examples = []
                for text, annotations in batch:
                    # create Example
                    # doc = nlp(text)
                    doc = nlp.make_doc(text)
                    examples.append(Example.from_dict(doc, annotations))

                # Update the model
                nlp.update(examples, losses=losses, drop=0.3)
                print(losses)

        # doc = nlp("book my a ticket from Boston to atlnta for this tursday")
        # print([(ent.text, ent.label_) for ent in doc.ents])

        # Now save the model to disk
        nlp.to_disk("./flight.model")

        # test
        self.custom_nlp = nlp

    def process(self, text):
        if self.custom_nlp is None:
            return None

        lower_text = text.lower()
        doc = self.custom_nlp(lower_text)

        return [Entity(ent_value=ent.text, ent_type=ent.label_, start_index=ent.start_char, end_index=ent.end_char)
                for ent in doc.ents]

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
                    if gt_entity.start_index == pred_entity.start_index and \
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

    nlp_engine = NLPEngine(nlp=spacy.load("./flight.model"))

    # read dataset
    _dataset = Dataset.read_dataset("dataset.json")

    # train model
    # nlp_engine.train(_dataset)

    # test expression
    res = nlp_engine.process("book my a ticket from Boston to atlnta for this tursday")
    for ent in res:
        print(ent.ent_value, ent.ent_type, ent.start_index, ent.end_index)

    # validate the solution ?
    nlp_engine.evaluate_accuracy(_dataset)

    # true_positive: 10856
    # false_negative: 6
    # false_positive: 6
    # Total Score: 0.9989
