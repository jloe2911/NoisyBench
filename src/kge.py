import torch.nn as nn
import torch as th
from pykeen.models import TransE, DistMult, ConvKB, TransD, ConvE, RotatE

class OrderE(TransE):
    def __init__(self, *args, **kwargs):
        super(OrderE, self).__init__(*args, **kwargs)

    def forward(self, h_indices, r_indices, t_indices, mode = None):
        h, _, t = self._get_representations(h=h_indices, r=r_indices, t=t_indices, mode=mode)
        order_loss = th.linalg.norm(th.relu(t-h), dim=1)
        return -order_loss

    def score_hrt(self, hrt_batch, mode = None):
        h, r, t = self._get_representations(h=hrt_batch[:, 0], r = hrt_batch[:, 1], t=hrt_batch[:, 2], mode=mode)
        return -th.linalg.norm(th.relu(t-h), dim=1)

    def distance(self, hrt_batch, mode = None):
        h, r, t = self._get_representations(h=hrt_batch[:, 0], r = hrt_batch[:, 1], t=hrt_batch[:, 2], mode=mode)
        distance = th.linalg.norm(t-h, dim=1)
        return  -distance

class KGEModule(nn.Module):
    def __init__(self, kge_model, triples_factory, embedding_dim, random_seed):
        super().__init__()
        self.triples_factory = triples_factory
        self.embedding_dim = embedding_dim
        self.random_seed = random_seed
        self.kge_model = kge_model

        if kge_model == "transe":
            self.kg_module =  TransE(triples_factory=self.triples_factory,
                                     embedding_dim=self.embedding_dim,
                                     scoring_fct_norm=2,
                                     random_seed = self.random_seed)
        elif kge_model == 'transd':
            self.kg_module = TransD(triples_factory=self.triples_factory,
                                    embedding_dim=self.embedding_dim,
                                    random_seed = self.random_seed)
        elif kge_model == "distmult":
            self.kg_module = DistMult(triples_factory=self.triples_factory,
                                     embedding_dim=self.embedding_dim,
                                      random_seed = self.random_seed,
                                      regularizer= None)
        elif kge_model == "convkb":
            self.kg_module = ConvKB(triples_factory=self.triples_factory,
                                     embedding_dim=self.embedding_dim,
                                    num_filters=10,
                                     random_seed = self.random_seed)
        elif kge_model == "conve":
            self.kg_module = ConvE(triples_factory=self.triples_factory,
                                   embedding_dim=self.embedding_dim,
                                   output_channels=10,
                                   embedding_height=16,
                                   embedding_width=16,
                                   random_seed = self.random_seed)
        elif kge_model == "ordere":
            self.kg_module =  OrderE(triples_factory=self.triples_factory,
                                     embedding_dim=self.embedding_dim,
                                     scoring_fct_norm=2,
                                     random_seed = self.random_seed)
        
        elif kge_model == "rotate":
            self.kg_module = RotatE(triples_factory=self.triples_factory,
                                        embedding_dim=self.embedding_dim,
                                        random_seed = self.random_seed)
            
    def forward(self, data):
        h, r, t = data
        logits = self.kg_module.forward(h, r, t, mode=None)
        if self.kge_model in ["distmult"]:
            logits = logits
        return logits

    def predict(self, data):
        h, r, t = data
        batch_hrt = th.stack([h,r,t], dim=1)
        logits = self.kg_module.score_hrt(batch_hrt)
        return logits

    def distance(self, data):
        h, r, t = data
        batch_hrt = th.stack([h,r,t], dim=1)
        dist = self.kg_module.distance(batch_hrt)
        return dist