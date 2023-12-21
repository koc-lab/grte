from math import log
from typing import List

from tqdm.auto import tqdm


class PPMI:
    def __init__(self, doc_content_list: List[str], window_size: int = 20, min_count=1):
        self.doc_content_list = doc_content_list
        self.window_size = window_size
        self.min_count = min_count

        self.vocab, self.vocab_size, self.word_freq = self.build_vocab()
        self.word_id_map = self.build_word_id_map()

        self.windows = self.build_windows()
        self.word_window_freq = self.build_word_window_freq()

        self.word_pair_count = self.build_word_pair_count_dict()
        (self.weight_ff, self.row_ff, self.col_ff) = self.build_ppmi()

    def build_vocab(self):
        word_freq = {}

        # Count word occurrences
        for docs in self.doc_content_list:
            words = docs.split()
            for word in words:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1

        # Filter words based on min_count
        word_freq = {word: count for word, count in word_freq.items() if count >= self.min_count}

        vocab, vocab_size = list(word_freq.keys()), len(word_freq)
        return vocab, vocab_size, word_freq

    def build_word_id_map(self):
        word_id_map = {}
        for i in range(len(self.vocab)):
            word_id_map[self.vocab[i]] = i
        return word_id_map

    def build_windows(self):
        windows = []
        for doc_words in tqdm(self.doc_content_list):
            words = doc_words.split()
            words = [word for word in words if word in self.vocab]
            if len(words) <= self.window_size:
                windows.append(words)
            else:
                for j in range(len(words) - self.window_size + 1):
                    window = words[j : j + self.window_size]
                    windows.append(window)
        return windows

    def build_word_window_freq(self):
        word_window_freq = {}  # w(i) ids: original words
        for window in self.windows:
            appeared = set()
            for _, word in enumerate(window):
                if word in appeared:
                    continue
                if word in word_window_freq:
                    word_window_freq[word] += 1
                else:
                    word_window_freq[word] = 1
                appeared.add(word)

        return word_window_freq

    def build_word_pair_count_dict(self):
        word_pair_count = {}  # w(i,j) ids: word ids (can be looked from word_id_map)
        for window in tqdm(self.windows):
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_i_id = self.word_id_map[word_i]
                    word_j = window[j]
                    word_j_id = self.word_id_map[word_j]

                    if word_i_id == word_j_id:
                        continue

                    word_pair_str = str(word_i_id) + "," + str(word_j_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    # two orders
                    word_pair_str = str(word_j_id) + "," + str(word_i_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
        return word_pair_count

    def build_ppmi(self):
        row_ff, col_ff, weight_ff = [], [], []
        num_window = len(self.windows)

        for key in self.word_pair_count:
            temp = key.split(",")
            i = int(temp[0])
            j = int(temp[1])
            count = self.word_pair_count[key]
            word_freq_i = self.word_window_freq[self.vocab[i]]
            word_freq_j = self.word_window_freq[self.vocab[j]]
            pmi = log((1.0 * count / num_window) / (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
            if pmi <= 0:
                continue

            row_ff.append(i)
            col_ff.append(j)
            weight_ff.append(pmi)
        return (weight_ff, row_ff, col_ff)


class TFIDF:
    def __init__(self, word_id_map, doc_content_list, vocab, word_freq):
        self.word_id_map = word_id_map
        self.doc_content_list = doc_content_list
        self.vocab = vocab
        self.word_freq = word_freq
        self.doc_word_freq = self.build_doc_word_freq()
        self.weight_nf, self.row_nf, self.col_nf = self.build_tfidf()

    def build_doc_word_freq(self):
        doc_word_freq = {}
        for doc_id, doc_words in enumerate(self.doc_content_list):
            words = doc_words.split()
            words = [word for word in words if word in self.vocab]
            for word in words:
                word_id = self.word_id_map[word]
                doc_word_str = str(doc_id) + "," + str(word_id)
                if doc_word_str in doc_word_freq:
                    doc_word_freq[doc_word_str] += 1
                else:
                    doc_word_freq[doc_word_str] = 1
        return doc_word_freq

    def build_tfidf(self):
        row_nf, col_nf, weight_nf = [], [], []

        for i, docs in tqdm(enumerate(self.doc_content_list)):
            words = docs.split()
            words = [word for word in words if word in self.vocab]
            doc_word_set = set()
            for word in words:
                if word in doc_word_set:
                    continue
                j = self.word_id_map[word]
                key = str(i) + "," + str(j)
                freq = self.doc_word_freq[key]

                row_nf.append(i)
                col_nf.append(j)
                idf = log(1.0 * len(self.doc_content_list) / self.word_freq[self.vocab[j]])

                weight_nf.append(freq * idf + 1e-6)
                doc_word_set.add(word)
        return weight_nf, row_nf, col_nf
