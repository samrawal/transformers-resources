import pickle
import logging
import os

from transformers.file_utils import is_tf_available
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from transformers.data.metrics import *

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


class CustomSTSProcessor(DataProcessor):
    """Processor for the Custom STS data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            '/home/scrawal/MS/qasc/siagen/sia_traindev.p', "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            '/home/scrawal/MS/qasc/siagen/sia_traindev.p', "dev")

    def get_file_examples(self, data_dir, filename, label='data'):
        """Same as get_{train,dev}_examples, but with a specific file"""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, filename)), label)

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, path, set_type):
        """Creates examples for the training and dev sets."""
        train, dev = pickle.load(open(path, 'rb'))
        if set_type == 'train':
            lines = train
        else:
            lines = dev

        examples = []    
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[0]
            text_b = line[1]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

custom_tasks_num_labels = {
    "customsts": 1,
}

custom_processors = {
    "customsts": CustomSTSProcessor,
}

custom_output_modes = {
    "customsts": "regression",
}

# Copied from `transformers.data.metrics`; add custom metrics here
def custom_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "customsts":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)
