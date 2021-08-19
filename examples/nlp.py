import argparse
import csv
import os
import sys

import torch
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import TensorDataset

project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)
from src import scaling
from src.training import *


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

		Args:
			guid: Unique id for the example.
			text_a: string. The untokenized text of the first sequence. For single
			sequence tasks, only this sequence must be specified.
			text_b: (Optional) string. The untokenized text of the second sequence.
			Only must be specified for sequence pair tasks.
			label: (Optional) string. The label of the example. This should be
			specified for train and dev examples, but not for test examples.
		"""
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        # logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid,
                             text_a=text_a,
                             text_b=text_b,
                             label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid,
                             text_a=text_a,
                             text_b=None,
                             label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid,
                             text_a=text_a,
                             text_b=None,
                             label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        # if ex_index < 5:
        # 	logger.info("*** Example ***")
        # 	logger.info("guid: %s" % (example.guid))
        # 	logger.info("tokens: %s" % " ".join(
        # 			[str(x) for x in tokens]))
        # 	logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        # 	logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        # 	logger.info(
        # 			"segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        # 	logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',
                        default=0.0005,
                        type=float,
                        help='learning rate')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--resume',
                        '-r',
                        action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--epoch_size', type=int)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument(
        '--model',
        type=str,
        help=
        'bert-base-uncased/bert-large-uncased/bert-base-cased/bert-large-cased/'
        'bert-base-multilingual-uncased/bert-base-multilingual-cased')
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--task_name', type=str, help='cola/sst2/mrpc')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--target', type=float)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--manager_addr', type=str, default='localhost')
    parser.add_argument('--manager_port', type=int, default=17834)
    parser.add_argument('--job_id', type=int)
    parser.add_argument('--port', type=int)
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


ARGS = get_args()

device = 'cuda'
torch.backends.cudnn.benchmark = True

sa = scaling.ScalingAgent(ARGS.job_id, 'nlp', ARGS.local_rank, ARGS.port,
                          ARGS.manager_addr, ARGS.manager_port)

# Data
print('==> Preparing data..')

processors = {
    "cola": ColaProcessor,
    "sst2": Sst2Processor,
    "mrpc": MrpcProcessor,
}

num_labels_task = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
}

task_name = ARGS.task_name.lower()

processor = processors[task_name]()
num_labels = num_labels_task[task_name]
label_list = processor.get_labels()

tokenizer = BertTokenizer.from_pretrained(ARGS.model, do_lower_case=True)

train_examples = processor.get_train_examples(ARGS.data_dir)
train_features = convert_examples_to_features(train_examples, label_list,
                                              ARGS.max_seq_length, tokenizer)
all_input_ids = torch.tensor([f.input_ids for f in train_features],
                             dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features],
                              dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features],
                               dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features],
                             dtype=torch.long)
trainset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                         all_label_ids)
if ARGS.epoch_size is not None and ARGS.epoch_size < len(trainset):
    trainset = torch.utils.data.Subset(trainset, range(ARGS.epoch_size))

eval_examples = processor.get_dev_examples(ARGS.data_dir)
eval_features = convert_examples_to_features(eval_examples, label_list,
                                             ARGS.max_seq_length, tokenizer)
all_input_ids = torch.tensor([f.input_ids for f in eval_features],
                             dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features],
                              dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features],
                               dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in eval_features],
                             dtype=torch.long)
testset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                        all_label_ids)

print('==> Loaded %d training samples, %d test samples' %
      (len(trainset), len(testset)))

# Model
print('==> Building model: ' + ARGS.model)
net = BertForSequenceClassification.from_pretrained(
    ARGS.model,
    # cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(ARGS.local_rank),
    num_labels=num_labels)

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if ARGS.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(
        ARGS.model_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(ARGS.model_dir + '/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']

net = net.to(device)

param_optimizer = list(net.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [{
    'params':
    [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    'weight_decay':
    0.01
}, {
    'params':
    [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    'weight_decay':
    0.0
}]
optimizer = BertAdam(optimizer_grouped_parameters, lr=ARGS.lr)
criterion = torch.nn.CrossEntropyLoss()

sa.load(net=net,
        criterion=criterion,
        optimizer=optimizer,
        trainset=trainset,
        testset=testset,
        batch_size=ARGS.batch_size,
        lr=ARGS.lr,
        num_labels=num_labels,
        start_epoch=start_epoch,
        scale=True if ARGS.scale else False)

if ARGS.resume:
    sa.adjust_learning_rate(checkpoint['lr'])

epoch = train(sa, patience=ARGS.patience)
save(sa, epoch, ARGS.model_dir)
