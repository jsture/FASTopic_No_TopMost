import torch
from torch.utils.data import DataLoader
import numpy as np
from fastopic_no_topmost.topmost.preprocessing import Preprocessing
from fastopic_no_topmost._utils import DocEmbedModel


class RawDataset:
    def __init__(
        self,
        docs,
        preprocessing=None,
        batch_size=200,
        device="cpu",
        as_tensor=True,
        contextual_embed=False,
        pretrained_WE=True,
        doc_embed_model="all-MiniLM-L6-v2",
        embed_model_device=None,
        verbose=False,
    ):
        if preprocessing is None:
            preprocessing = Preprocessing(verbose=verbose)

        rst = preprocessing.preprocess(docs, pretrained_WE=pretrained_WE)
        self.train_data = rst["train_bow"]
        self.train_texts = rst["train_texts"]
        self.vocab = rst["vocab"]

        self.vocab_size = len(self.vocab)

        if contextual_embed:
            if embed_model_device is None:
                embed_model_device = device

            if isinstance(doc_embed_model, str):
                self.doc_embedder = DocEmbedModel(
                    doc_embed_model, embed_model_device, verbose=verbose
                )
            else:
                self.doc_embedder = doc_embed_model

            self.train_contextual_embed = self.doc_embedder.encode(docs)
            self.contextual_embed_size = self.train_contextual_embed.shape[1]

        if as_tensor:
            if contextual_embed:
                self.train_data = np.concatenate(
                    (self.train_data, self.train_contextual_embed), axis=1
                )

            self.train_data = torch.from_numpy(self.train_data).float().to(device)
            self.train_dataloader = DataLoader(
                self.train_data, batch_size=batch_size, shuffle=True
            )
