import re

words_path = '/iesl/canvas/nnayak/data/box_ner/conll/eng.train'
train_sentences_file = words_path

TAGS = [
    'O', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER',
         'B-LOC', 'B-MISC', 'B-ORG', 'B-PER']
__UNK = "__UNK"
__PAD = "__PAD"


def get_vocab(vocab_path):
  with open(vocab_path) as f:
    words = [__UNK, __PAD]
    words += [l.lower().strip() for l in f]
    return {word: idx for idx, word in enumerate(words)}

train_sentences = []
train_labels = []

vocab = get_vocab('/iesl/canvas/nnayak/data/box_ner/glove/glove.6B.vocab')
with open(train_sentences_file) as f:
  unk_idx = vocab[__UNK]
  for sentence in f.read().splitlines():
    s = [vocab[token] if token in vocab
         else unk_idx
         for token in sentence.split(' ')]
    train_sentences.append(s)

lower_re = re.compile("^[a-z]+$")
upper_re = re.compile("^[A-Z]+$")
init_caps_re = re.compile("^[A-Z][a-z]+$")
one_upper_re = re.compile("^[a-z]*[A-Z][a-z]*$")

SHAPE_RES = [lower_re, upper_re, init_caps_re, one_upper_re]

def get_shape_idx(word):
  for i, shape_re in enumerate(SHAPE_RES):
    if shape_re.match(word):
      return i
  return len(SHAPE_RES)


with open(train_sentences_file) as f:
  _, _ = f.readline(), f.readline() # some stuff at the start

  current_sentence = []
  sentences = []

  for line in f.readlines()[:100]:
    fields = line.strip().split()
    if len(fields) == 0:
      words, tags, shapes = zip(*current_sentence)
      word_idxs = [vocab[word] if word in vocab else unk_idx
          for word in words]
      sentences.append((word_idxs, tags, shapes))
      current_sentence = []
    else:
      assert len(fields) == 4
      word, _, _, tag = fields
      tag_idx = TAGS.index(tag)
      shape_idx = get_shape_idx(word)
      current_sentence.append((word.lower(), tag_idx, shape_idx))


class NERDataset(torch.Dataset):
  def __init__(self, sentences, tags, shapes):
    self.examples = zip([sentences, tags, shapes])

  def __len__():
    return len(self.examples)

  def __getitem__(i):
    return self.examples[i]


class NERDataloader(torch.Dataloader):

dataloader = Dataloader(dataset, batch_size, shuffle=True, num_workers=0)
