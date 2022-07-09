import pandas as pd

import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


spacy_eng = spacy.load('en_core_web_sm')
# map rate to sentiment: 1, 2 -> negative, 3 -> natural, 4, 5 -> positive
rate_senti_map = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}


class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        # PAD->padding, SOS->Start of Sequence, EOS->End of Sequence, UNK->Unknown word did not shown in train set
        self.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi['<UNK>']
            for token in tokenized_text
        ]


class ReviewDataset(Dataset):
    def __init__(self, root_dir, freq_threshold=5, task='Sentiment'):
        super(ReviewDataset, self).__init__()
        self.root_dir = root_dir
        self.vocab = Vocabulary(freq_threshold)
        self.task = task
        self.df = pd.read_csv(root_dir + 'reviews.csv', index_col='Id')
        self.text = self.df['Review']

        self.vocab.build_vocabulary(self.text.tolist())

    def __getitem__(self, idx):
        if self.task == 'Sentiment':
            rate = self.df.loc[idx, 'Label']
            text = self.df.loc[idx, 'Review']

            numericalized_caption = [self.vocab.stoi["<SOS>"]]
            numericalized_caption += self.vocab.numericalize(text)
            numericalized_caption.append(self.vocab.stoi["<EOS>"])

            sentiment = rate_senti_map[rate]
            return torch.tensor(numericalized_caption), torch.tensor(sentiment)

        elif self.task == 'Topic':
            text = self.df.loc[idx, 'Review']

            numericalized_caption = [self.vocab.stoi["<SOS>"]]
            numericalized_caption += self.vocab.numericalize(text)
            numericalized_caption.append(self.vocab.stoi["<EOS>"])

            return torch.tensor(numericalized_caption)

        else:
            raise 'Wrong Task! Please select Sentiment or Topic'

    def __len__(self):
        return self.df.shape[0]


class MyCollate:
    def __init__(self, pad_idx, task):
        self.pad_idx = pad_idx
        self.task = task

    def __call__(self, batch):
        # Sentiment task return (texts, rates)
        if self.task == 'Sentiment':
            texts = [item[0] for item in batch]
            texts = pad_sequence(texts, batch_first=True, padding_value=self.pad_idx)

            rates = [item[1].unsqueeze(0) for item in batch]
            rates = torch.cat(rates, dim=0)
            return texts, rates

        # Topic task return texts only
        elif self.task == 'Topic':
            texts = [item for item in batch]
            texts = pad_sequence(texts, batch_first=True, padding_value=self.pad_idx)

            return texts


if __name__ == '__main__':
    Task = 'Topic'
    print('Generating dataloader for task:{}...'.format(Task))
    dataset = ReviewDataset(root_dir='data/', task=Task)
    pad_idx = dataset.vocab.stoi['<PAD>']
    loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, collate_fn=MyCollate(pad_idx=pad_idx, task=Task))
    for idx, reviews in enumerate(loader):
        print(reviews.shape)






