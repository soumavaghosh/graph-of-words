import re
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import stopwords
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer

stop_words = list(set(stopwords.words('english')))
rem = [',', '.', '?', ':', '...', '-', '"', "'", '!', "'s", "'nt", "'m", "'ve", "'d", '``', '\'\'', "'re", '(', ')', 'n\'t']
stop_words.extend(rem)

with open('../amazonreviews/train.ft.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()

with open('../amazonreviews/test.ft.txt', 'r', encoding='utf-8') as f:
    data.extend(f.readlines())

print('data read')

label = ['0' if x.startswith('__label__1 ') else '1' for x in data]
data = [x[11:] for x in data]

with open('../amazonreviews/amazon_graph_indicator.txt', 'w') as f:
    f.write('\n'.join(label))

def text_cleaner(text):
    if 'www.' in text or 'http:' in text or 'https:' in text or '.com' in text:
        text = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", text)
    text = sent_tokenize(text)
    text = [word_tokenize(x.lower()) for x in text]
    out = []
    for j in range(len(text)):
        text[j] = pos_tag(text[j])
        text[j] = [PorterStemmer().stem(x[0]) for x in text[j] if x[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ']]
        text[j] = [x for x in text[j] if x not in stop_words]
        out.extend(text[j])
    return out

print(data[16])
print(text_cleaner(data[16]))
