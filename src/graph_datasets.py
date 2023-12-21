import pickle as pkl
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from constants import get_label_to_int
from graph_stat import PPMI, TFIDF  # noqa
from scipy.sparse import csr_matrix as csr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import kneighbors_graph as knn


class Dataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        path1 = Path.joinpath(Path(__file__).parent.parent, "data-raw", dataset_name + ".txt")
        path2 = Path.joinpath(Path(__file__).parent.parent, "data-raw/corpus", dataset_name + ".clean.txt")

        self.train_ids, self.test_ids, self.y = self.foo(path1)
        #! this y is raw labels, they are strings
        self.doc_list = self.boo(path2)
        self.transform_variables()
        self.generate_graphs()

        self.label_to_int = get_label_to_int(dataset_name)
        self.y = torch.tensor([self.label_to_int[label] for label in self.y]).to("mps")

    def generate_graphs(self):
        min_count = 1 if self.dataset_name != "20ng" else 20
        ppmi = PPMI(self.doc_list, window_size=20, min_count=min_count)
        vectorizer = TfidfVectorizer(vocabulary=ppmi.vocab)
        tfidf_matrix = vectorizer.fit_transform(self.doc_list)
        self.NF_csr = csr(tfidf_matrix)

        # tfidf = TFIDF(ppmi.word_id_map, self.doc_list, ppmi.vocab, ppmi.word_freq)
        # w_nf, r_nf, c_nf = tfidf.weight_nf, tfidf.row_nf, tfidf.col_nf
        w_ff, r_ff, c_ff = ppmi.weight_ff, ppmi.row_ff, ppmi.col_ff

        # self.NF_csr = csr(
        #     (w_nf, (r_nf, c_nf)), shape=(len(self.doc_list), ppmi.vocab_size)
        # )
        self.FF_csr = csr((w_ff, (r_ff, c_ff)), shape=(ppmi.vocab_size, ppmi.vocab_size))

        k = 25 if self.dataset_name != "20ng" else 100

        self.NN_csr = knn(
            self.NF_csr,
            n_neighbors=k,
            metric="minkowski",
            mode="distance",
            include_self=True,
        )

        self.FN_csr = self.NF_csr.T

        self.FF = Dataset.to_torch(self.FF_csr)
        self.NF = Dataset.to_torch(self.NF_csr)
        self.NN = Dataset.to_torch(self.NN_csr)
        self.FN = Dataset.to_torch(self.FN_csr)

    def foo(self, path):
        y, train_ids, test_ids = [], [], []
        with open(path) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                y.append(line.strip().split("\t")[2])
                temp = line.split("\t")
                if temp[1].find("test") != -1:
                    test_ids.append(i)
                elif temp[1].find("train") != -1:
                    train_ids.append(i)

        return train_ids, test_ids, y

    def boo(self, path):
        doc_content_list = []
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                doc_content_list.append(line.strip())
        return doc_content_list

    def transform_variables(self):
        self.y = np.array(self.y)
        self.doc_list = np.array(self.doc_list)
        self.train_ids = np.array(self.train_ids)
        self.test_ids = np.array(self.test_ids)

    @staticmethod
    def to_torch(M: sp.csr_matrix):
        M = M.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
        values = torch.from_numpy(M.data)
        shape = torch.Size(M.shape)
        T = torch.sparse.FloatTensor(indices, values, shape)  # type: ignore

        return T

    def __repr__(self) -> str:
        print_str = (
            f"RawDataset({self.dataset_name})"
            f"\nTotal  Number of documents: {len(self.doc_list)}"
            f"\nNumber of initial training documents: {len(self.train_ids)}"
            f"\nNumber of initial test documents: {len(self.test_ids)}"
        )
        return print_str


if __name__ == "__main__":
    # To generate the dataset files, run this script.
    # for dataset_name in ["mr", "ohsumed", "R8", "R52", "20ng"]:
    for dataset_name in ["20ng"]:
        # for dataset_name in ["mr"]:
        DATA_DIR = Path(__file__).parent.parent.joinpath("data-processed")
        DATA_DIR.mkdir(exist_ok=True)
        dataset_path = DATA_DIR.joinpath(dataset_name + ".pkl")
        if not dataset_path.exists():
            dataset = Dataset(dataset_name=dataset_name)
            with open(dataset_path, "wb") as f:
                pkl.dump(dataset, f)
        else:
            print(f"{dataset_path} already exists.")

    # @staticmethod
    # def max_min_normalize_graph(x):
    #     x_normed = (x - x.min(0, keepdim=True)[0]) / (
    #         x.max(0, keepdim=True)[0] - x.min(0, keepdim=True)[0]
    #     )
