import re
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import stopwords
from tqdm import tqdm
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup

class InputData:
    def __init__(self, data, label):
        self.stop_words = list(set(stopwords.words('english')))
        rem = [',', '.', '?', ':', '...', '-', '"', "'", '!', "'s", "'nt", "'m", "'ve", "'d", '``', '\'\'', "'re", '(',')','n\'t']
        self.stop_words.extend(rem)
        self.label = label
        self.data = self.text_cleaner(data)
        self.words = set(self.data)

    def text_cleaner(self, text):
        if 'www.' in text or 'http:' in text or 'https:' in text or '.com' in text:
            text = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", text)
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub('\[[^]]*\]', '', text)
        pattern = r'[^a-zA-z0-9\s.!?]'
        text = sent_tokenize(text)
        out = []
        for j in range(len(text)):
            text[j] = re.sub(pattern, '', text[j])
            text[j] = word_tokenize(text[j])
            text[j] = [x.lower() for x in text[j]]
            text[j] = pos_tag(text[j])
            text[j] = [PorterStemmer().stem(x[0]) for x in text[j] if x[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ']]
            text[j] = [x for x in text[j] if x not in self.stop_words]
            out.extend(text[j])
        return out



