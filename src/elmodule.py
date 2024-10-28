import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn

from itertools import cycle

import mowl
mowl.init_jvm('10g')
from mowl.base_models import EmbeddingELModel
from mowl.nn import ELEmModule, ELBoxModule, BoxSquaredELModule
from mowl.utils.data import FastTensorDataLoader

from src.utils import *

class ELModule(nn.Module):
    def __init__(self, module_name, dim, nb_classes, nb_individuals, nb_roles):
        super().__init__()
        self.module_name = module_name
        self.dim = dim
        self.nb_classes = nb_classes
        self.nb_individuals = nb_individuals
        self.nb_roles = nb_roles
        self.set_module(self.module_name)
        self.indiv_embeddings = nn.Embedding(self.nb_individuals, self.dim)

    def set_module(self, module_name):
        if module_name == "elem":
            self.module = ELEmModule(self.nb_classes, self.nb_roles, embed_dim = self.dim)
        elif module_name == "elbox":
            self.module = ELBoxModule(self.nb_classes, self.nb_roles, embed_dim = self.dim)
        elif module_name == "box2el":
            self.module = BoxSquaredELModule(self.nb_classes, self.nb_roles, embed_dim = self.dim)

    def tbox_forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def abox_forward(self, indiv_idxs, mode):
        class_embed = self.module.class_center if self.module_name == "box2el" else self.module.class_embed
        indiv_embed = self.indiv_embeddings(indiv_idxs)
        all_class_embed = class_embed.weight
        all_indiv_embed = self.indiv_embeddings.weight
        if 'link_prediction' in mode:
            score = torch.mm(indiv_embed, all_indiv_embed.t())
        else: 
            score = torch.mm(indiv_embed, all_class_embed.t())
            if self.module_name == "elem":
                rad_embed = self.module.class_rad.weight
                rad_embed = torch.abs(rad_embed).view(1, -1)
                score = score + rad_embed
            elif self.module_name in ["elbox", "box2el"]:
                offset_embed = self.module.class_offset.weight
                offset_embed = torch.abs(offset_embed).mean(dim=1).view(1, -1)
                score = score + offset_embed
        return score

class ElModel(EmbeddingELModel):
    def __init__(self, dataset, module_name, dim, margin, batch_size, test_batch_size, epochs, learning_rate, device):
        self.module_name = module_name
        self.dim = dim
        self.margin = margin
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.nb_classes = len(dataset.classes)
        self.nb_individuals = len(dataset.individuals)
        self.nb_roles = len(dataset.object_properties)
        self.module = ELModule(self.module_name, self.dim, self.nb_classes, self.nb_individuals, self.nb_roles)
        super().__init__(dataset=dataset, embed_dim=dim, batch_size=batch_size, device=device)

    def get_abox_data(self, dataset_type):
        if dataset_type == "train":
            ontology = self.dataset.ontology
        elif dataset_type == "valid":
            ontology = self.dataset.validation
        elif dataset_type == "test":
            ontology = self.dataset.testing
        
        abox = []
        property_assertions = []
        
        for cls in self.dataset.classes:
            abox.extend(list(ontology.getClassAssertionAxioms(cls)))  

        object_properties = list(ontology.getObjectPropertiesInSignature())
        individuals = ontology.getIndividualsInSignature()
        for individual in individuals:
            for prop in object_properties:
                assertions = ontology.getObjectPropertyAssertionAxioms(individual)
                for assertion in assertions:
                    if assertion.getProperty().equals(prop):  
                        property_assertions.append(assertion)

        nb_individuals = len(self.dataset.individuals)
        nb_classes = len(self.dataset.classes)
        
        owl_indiv_to_id = self.dataset.individuals.to_index_dict()
        owl_class_to_id = self.dataset.classes.to_index_dict()
        
        labels = np.zeros((nb_individuals, nb_classes), dtype=np.int32)
        for axiom in abox:
            cls = axiom.getClassExpression()
            indiv = axiom.getIndividual()
            cls_id = owl_class_to_id[cls]
            indiv_id = owl_indiv_to_id[indiv]
            labels[indiv_id, cls_id] = 1
        
        property_labels = np.zeros((nb_individuals, nb_individuals), dtype=np.int32)
        for axiom in property_assertions:
            subject = axiom.getSubject()
            obj = axiom.getObject()
            subj_id = owl_indiv_to_id[subject]
            obj_id = owl_indiv_to_id[obj]
            property_labels[subj_id, obj_id] = 1  
        
        idxs = np.arange(nb_individuals)
        
        return torch.tensor(idxs), torch.FloatTensor(labels), torch.FloatTensor(property_labels)
                                                                  
    def _train(self):
        abox_ds_train = self.get_abox_data("train")
        abox_dl_train = FastTensorDataLoader(*abox_ds_train, batch_size=self.batch_size, shuffle=True)

        el_dls = {gci_name: DataLoader(ds, batch_size=self.batch_size, shuffle=True) for gci_name, ds in self.training_datasets.items() if len(ds) > 0}
        el_dls_sizes = {gci_name: len(ds) for gci_name, ds in self.training_datasets.items() if len(ds) > 0}

        if len(self.dataset.individuals) > el_dls_sizes["gci0"]:
            main_dl = abox_dl_train
            main_dl_name = "abox"
        else:
            main_dl = el_dls["gci0"]
            main_dl_name = "gci0"

        total_el_dls_size = sum(el_dls_sizes.values())
        el_dls_weights = {gci_name: ds_size / total_el_dls_size for gci_name, ds_size in el_dls_sizes.items()}

        if main_dl_name == "gci0":
            el_dls = {gci_name: cycle(dl) for gci_name, dl in el_dls.items() if gci_name != "gci0"}
            abox_dl_train = cycle(abox_dl_train)
        else:
            el_dls = {gci_name: cycle(dl) for gci_name, dl in el_dls.items()}

        optimizer = torch.optim.Adam(self.module.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            module = self.module.to(self.device)
            module.train()

            train_el_loss = 0
            train_abox_loss = 0
            train_abox_loss_lp = 0

            for batch_data in main_dl:
                if main_dl_name == "abox":
                    ind_idxs, labels, propery_labels = batch_data
                    gci0_batch = next(el_dls["gci0"]).to(self.device)
                elif main_dl_name == "gci0":
                    ind_idxs, labels, propery_labels = next(abox_dl_train)
                    gci0_batch = batch_data.to(self.device)

                pos_gci0 = module.tbox_forward(gci0_batch, "gci0").mean() * el_dls_weights["gci0"]
                neg_idxs = np.random.choice(self.nb_classes, size=len(gci0_batch), replace=True)
                neg_batch = torch.tensor(neg_idxs, dtype=torch.long, device=self.device)
                neg_data = torch.cat((gci0_batch[:, :1], neg_batch.unsqueeze(1)), dim=1)
                neg_gci0 = module.tbox_forward(neg_data, "gci0").mean() * el_dls_weights["gci0"]

                el_loss = -F.logsigmoid(-pos_gci0 + neg_gci0 - self.margin).mean()

                for gci_name, gci_dl in el_dls.items():
                    if gci_name == "gci0":
                        continue
                    gci_batch = next(gci_dl).to(self.device)
                    pos_gci = module.tbox_forward(gci_batch, gci_name).mean() * el_dls_weights[gci_name]
                    neg_idxs = np.random.choice(self.nb_classes, size=len(gci_batch), replace=True)
                    neg_batch = torch.tensor(neg_idxs, dtype=torch.long, device=self.device)
                    neg_data = torch.cat((gci_batch[:, :2], neg_batch.unsqueeze(1)), dim=1)
                    neg_gci = module.tbox_forward(neg_data, gci_name).mean() * el_dls_weights[gci_name]

                    el_loss += -F.logsigmoid(-pos_gci + neg_gci - self.margin).mean()

                abox_logits = module.abox_forward(ind_idxs.to(self.device), 'membership')
                abox_loss = F.binary_cross_entropy_with_logits(abox_logits, labels.to(self.device))

                abox_logits_lp = module.abox_forward(ind_idxs.to(self.device), 'link_prediction')
                abox_loss_lp = F.binary_cross_entropy_with_logits(abox_logits_lp, propery_labels.to(self.device))

                loss = el_loss + abox_loss + abox_loss_lp

                loss.backward()
                optimizer.step()
                train_el_loss += el_loss.item()
                train_abox_loss += abox_loss.item()
                train_abox_loss_lp += abox_loss_lp.item()

            train_el_loss /= len(main_dl)
            train_abox_loss /= len(main_dl)
            train_abox_loss_lp /= len(main_dl)
            if (epoch % 25) == 0:
                print(f"Epoch: {epoch}, Training: EL loss: {train_el_loss:.4f}, ABox loss (Membership): {train_abox_loss:.4f}, ABox loss (Link Prediction): {train_abox_loss_lp:.4f}")
            
    def predict(self, heads, tails, mode):
        aux = heads.to(self.device)
        num_heads = len(heads)
        if 'link_prediction' in mode:
            tail_ids = torch.arange(len(self.dataset.individuals)).to(self.device)
        else: 
            tail_ids = torch.arange(len(self.dataset.classes)).to(self.device)
        heads = heads.to(self.device)
        heads = heads.repeat(len(tail_ids), 1).T
        heads = heads.reshape(-1)
        eval_tails = tail_ids.repeat(num_heads)
        data = torch.stack((heads, eval_tails), dim=1)

        self.module.eval()
        self.module.to(self.device)

        if "subsumption" in mode:    
            predictions = -self.module.tbox_forward(data, "gci0")
        elif "membership" in mode:
            predictions = self.module.abox_forward(aux, mode)
            max_ = torch.max(predictions)
            predictions = predictions - max_
        elif "link_prediction" in mode:
            predictions = self.module.abox_forward(aux, mode)
            max_ = torch.max(predictions)
            predictions = predictions - max_
        predictions = predictions.reshape(-1, len(tail_ids))

        return predictions
    
    def _eval(self, mode):
        if "subsumption" in mode:
            ds = self.testing_datasets["gci0"][:]
            sub_class = ds[:, 0]
            super_class = ds[:, 1]
            eval_dl = FastTensorDataLoader(sub_class, super_class, batch_size=self.test_batch_size, shuffle=False)
        elif "membership" in mode:
            _, labels, _ = self.get_abox_data("test")
            ds = torch.nonzero(labels).squeeze() 
            individuals = ds[:, 0] 
            classes = ds[:, 1]
            eval_dl = FastTensorDataLoader(individuals, classes, batch_size=self.test_batch_size, shuffle=False)       
        elif "link_prediction" in mode:
            _, _, property_labels = self.get_abox_data("test")
            ds = torch.nonzero(property_labels).squeeze()  
            individuals = ds[:, 0]  
            individuals2 = ds[:, 1]  
            eval_dl = FastTensorDataLoader(individuals, individuals2, batch_size=self.test_batch_size, shuffle=False)     

        ranks = dict()
        rank_vals = []

        mrr = 0
        hits_at_1 = 0
        hits_at_5 = 0
        hits_at_10 = 0
        
        with torch.no_grad():
            for head_idxs, tail_idxs in eval_dl:
                predictions = self.predict(head_idxs, tail_idxs, mode=mode)
                for i, head in enumerate(head_idxs):
                    tail = tail_idxs[i]
                    preds = predictions[i]
                    orderings = torch.argsort(preds, descending=True)
                    rank = torch.where(orderings == tail)[0].item()
                    rank_vals.append(rank)
                    if rank not in ranks:
                        ranks[rank] = 0
                    ranks[rank] += 1

                    mrr += 1/(rank+1)
                    if rank < 1:
                        hits_at_1 += 1
                    if rank < 5:
                        hits_at_5 += 1
                    if rank < 10:
                        hits_at_10 += 1
                        
            mrr /= eval_dl.dataset_len
            hits_at_1 /= eval_dl.dataset_len
            hits_at_5 /= eval_dl.dataset_len
            hits_at_10 /= eval_dl.dataset_len

            print(f'MRR: {mrr:.3f}, Hits@1: {hits_at_1:.3f}, Hits@5: {hits_at_5:.3f}, Hits@10: {hits_at_10:.3f}')
            return (mrr, hits_at_1, hits_at_5, hits_at_10)