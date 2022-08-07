#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive 
drive.mount('/content/gdrive')


# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


csv_file = 'gdrive/My Drive/STAT8307/Project/reviews.csv'
df_original = pd.read_csv(csv_file)
df = df_original.copy()


# ### Setup

# In[ ]:


get_ipython().system('pip install -q -U watermark')


# In[ ]:


get_ipython().system('pip install -qq transformers')


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -p numpy,pandas,torch,transformers')


# In[ ]:


#@title Setup & Config
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 8307
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# In[ ]:


def to_sentiment(rating):
    rating = int(rating)
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else: 
        return 2

df['sentiment'] = df.Label.apply(to_sentiment)


# In[ ]:


class_names = ['negative', 'neutral', 'positive']


# In[ ]:


df.tail()


# In[ ]:


df.shape


# In[ ]:


df['sentiment'].value_counts()


# In[ ]:


# text preprocessing part
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

stemmer = WordNetLemmatizer()
porter = PorterStemmer()
STOPWORDS = set(stopwords.words('english'))

def preprocessingText(corpus, lowercase=True, rmPunctuation=True, rpURL=True, rpNumber=True, stemming=True, rmStopwords=True):
    """Input is assumed to be vector of documents"""
    documents = []
    for text in corpus:
        document = text
        
        # HYPERPARAMETER
        # Converting to Lowercase
        if lowercase:
            document = document.lower()

        # replace URL
        if rpURL:
            # replace URL
            document = re.sub(r'http\S+', 'url', document, flags=re.MULTILINE)

        # replace numbers
        if rpNumber:
            document = re.sub("\d+", "number", document)

        # remove all special characters including punctuation
        if rmPunctuation:
            # only keep word
            document = re.sub(r'\W', ' ', document)
            # remove all single characters
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
            # Remove single characters from the start
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # OTHER PREPROCESSING METHODS
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        
        # removing stopwords
        document = document.split()
        if rmStopwords:
            document = [word for word in document if word not in STOPWORDS]
        elif not rmStopwords:
            document = [word for word in document]

        if stemming:
            # Lemmatization
            document = [stemmer.lemmatize(word) for word in document]
            # stemming
            document = [porter.stem(word) for word in document]

        document = ' '.join(document)
        documents.append(document)
    return documents


# In[ ]:


df['text'] = preprocessingText(df.Review, stemming=False, rmStopwords=False)
print(df.tail().text)


# In[ ]:


# let's make the data balanced first
np.random.seed(8307)
positive_indices = df[df.sentiment == 2].index
random_indices = np.random.choice(positive_indices, 5071, replace=False)
positive_sample = df.loc[random_indices]
positive_sample


# In[ ]:


df2 = pd.concat([positive_sample, df[df['sentiment'] != 2]], verify_integrity=True)
df2['sentiment'].value_counts()


# In[ ]:


sns.countplot(df.Label)
plt.xlabel('review score');


# In[ ]:


ax = sns.countplot(df.sentiment)
plt.xlabel('review sentiment')
ax.set_xticklabels(class_names);


# In[ ]:


ax = sns.countplot(df2.sentiment)
plt.xlabel('review sentiment')
ax.set_xticklabels(class_names);


# Currently the dataset is somehow balanced.

# ### Data Preprocessing

# In[ ]:


PRE_TRAINED_MODEL_NAME = 'bert-base-cased'


# In[ ]:


tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


# In[ ]:


df2.info()


# In[ ]:


encoding = tokenizer.encode_plus(
  sample_txt,
  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
  return_token_type_ids=False,
  padding=True, # change the upper line to this as there will be warnings
  return_attention_mask=True,
  return_tensors='pt',  # Return PyTorch tensors
)

encoding.keys()


# #### Decide sequence length.

# In[ ]:


token_lens = []

for txt in df2.text:
    tokens = tokenizer.encode(txt, max_length=512)
    token_lens.append(len(tokens))


# In[ ]:


sns.distplot(token_lens)
plt.xlim([0, 300]);
plt.xlabel('Token count');


# In[ ]:


MAX_LEN = 250


# In[ ]:


class CourseReviewDataset(Dataset):

    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
          review,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',
        )

        return {
          'review_text': review,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }


# In[ ]:


df_train, df_test = train_test_split(df2, test_size=0.1, stratify=df2['sentiment'], random_state=RANDOM_SEED)


# In[ ]:


df_train.shape, df_test.shape


# In[ ]:


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = CourseReviewDataset(
    reviews=df.text.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=2
    )


# In[ ]:


# fine-tune
BATCH_SIZE = 16

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)


# In[ ]:


data = next(iter(train_data_loader))
data.keys()


# In[ ]:


print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)


# ### Sentiment Analysis

# In[ ]:


bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)


# In[ ]:


class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask,
          return_dict=False # need to add this line
        )
        output = self.drop(pooled_output)
        return self.out(output)


# In[ ]:


model = SentimentClassifier(len(class_names))
model = model.to(device)


# In[ ]:


input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

print(input_ids.shape) # batch size x seq length
print(attention_mask.shape) # batch size x seq length


# In[ ]:


F.softmax(model(input_ids, attention_mask), dim=1)


# ### Training

# In[ ]:


EPOCHS = 10

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)


# In[ ]:


def train_epoch(
    model, 
    data_loader, 
    loss_fn, 
    optimizer, 
    device, 
    scheduler, 
    n_examples
    ):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


# In[ ]:


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nhistory = defaultdict(list)\n\nfor epoch in range(EPOCHS):\n\n    print(f'Epoch {epoch + 1}/{EPOCHS}')\n    print('-' * 10)\n\n    train_acc, train_loss = train_epoch(\n    model,\n    train_data_loader,    \n    loss_fn, \n    optimizer, \n    device, \n    scheduler, \n    len(df_train)\n    )\n\n    print(f'Train loss {train_loss} accuracy {train_acc}')\n\n    history['train_acc'].append(train_acc.item())\n    history['train_loss'].append(train_loss)")


# In[ ]:


plt.plot(history['train_acc'], label='train accuracy')

plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);


# ### Evaluation

# In[ ]:


test_acc, _ = eval_model(
  model,
  test_data_loader,
  loss_fn,
  device,
  len(df_test)
)

test_acc.item()


# In[ ]:


print(f'Test accuracy {test_acc}')


# In[ ]:


def get_predictions(model, data_loader):
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
    for d in data_loader:
        texts = d["review_text"]
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)

        probs = F.softmax(outputs, dim=1)

        review_texts.extend(texts)
        predictions.extend(preds)
        prediction_probs.extend(probs)
        real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values


# In[ ]:


y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
  model,
  test_data_loader
)


# In[ ]:


print(classification_report(y_test, y_pred, target_names=class_names))


# In[ ]:


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment');

cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)

