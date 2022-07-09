import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud


df = pd.read_csv('reviews.csv')
# (107018, 3) -> ['Id', 'Review', 'Label']
print(df.head(5))
print(Counter(df.Label))  # show the distribution of rating

# text = ''.join(df.loc[:, 'Review'])
text = ''.join(df.query('Label>4').loc[:, 'Review'])


def wordcloudplot(text):
    wc = WordCloud(width=800, height=560, background_color='white', collocations=False, min_font_size=10).generate(text)
    plt.figure(figsize=(8, 5.6), facecolor=None)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.imshow(wc, interpolation="bilinear")
    plt.show()


wordcloudplot(text=text)

