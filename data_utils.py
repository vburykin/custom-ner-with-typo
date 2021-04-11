import json


class Entity:
    def __init__(self, ent_type, ent_value, start_index=-1, end_index=-1):
        self.ent_type = ent_type
        self.ent_value = ent_value
        self.start_index = start_index
        self.end_index = end_index

    def __str__(self):
        return f"{self.ent_type}: {self.ent_value} ({self.start_index},{self.end_index})"


class Example:
    def __init__(self, text, intent, entities=None):
        self.text = text
        self.intent = intent
        self.entities = entities if entities is not None else []

    def __str__(self):
        return f"{self.text} (intent:{self.intent}, entities:{len(self.entities)})"


class Dataset:
    def __init__(self, examples=None):
        self.examples = examples if not None else []

    @classmethod
    def read_dataset(cls, file):
        examples = []
        with open(file) as f:
            loaded_examples = json.load(f)
            for example in loaded_examples:
                entities = []
                for loaded_entity in example["entities"]:
                    entities.append(Entity(loaded_entity["type"], loaded_entity["value"], loaded_entity["start_index"],
                                           loaded_entity["end_index"]))
                examples.append(Example(example["text"], example["intent"], entities))
        return Dataset(examples)
