import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle as pkl
import os

import torch 
import torch.nn as nn

from pykeen.models import TransE, DistMult, ConvKB, ERModel
from pykeen.triples import TriplesFactory

import mowl
mowl.init_jvm('10g')
from mowl.owlapi.defaults import TOP, BOT
from mowl.projection import OWL2VecStarProjector
from mowl.datasets import PathDataset
from mowl.utils.data import FastTensorDataLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from src.utils import *
from src.kge import *

class GraphModel():
    def __init__(self,
                 file_name,
                 dataset_name,
                 kge_model,
                 emb_dim,
                 margin,
                 weight_decay,
                 batch_size,
                 lr,
                 num_negs,
                 test_batch_size,
                 epochs,
                 device,
                 seed,
                 initial_tolerance,
                 ):

        self.file_name = file_name
        self.dataset_name = dataset_name
        self.kge_model = kge_model
        self.emb_dim = emb_dim
        self.margin = margin
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.lr = lr
        self.num_negs = num_negs
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.device = device
        self.seed = seed
        self.initial_tolerance = initial_tolerance
                
        self._triples_factory = None
        
        self._train_graph = None
        self._valid_subsumption_graph = None
        self._valid_membership_graph = None
        self._test_subsumption_graph = None
        self._test_membership_graph = None
        
        self._model_path = None
        self._node_to_id = None
        self._relation_to_id = None
        self._id_to_node = None
        self._id_to_relation = None
        self._classes = None
        self._object_properties = None
        self._individuals = None
        self._classes_ids = None
        self._object_properties_ids = None
        self._individuals_ids = None
                                
        # print("Parameters:")
        # print(f"\tFile name: {self.file_name}")
        # print(f"\tKGE model: {self.kge_model}")
        # print(f"\tEmbedding dimension: {self.emb_dim}")
        # print(f"\tMargin: {self.margin}")
        # print(f"\tWeight decay: {self.weight_decay}")
        # print(f"\tBatch size: {self.batch_size}")
        # print(f"\tLearning rate: {self.lr}")
        # print(f"\tNumber of negatives: {self.num_negs}")
        # print(f"\tTest batch size: {self.test_batch_size}")
        # print(f"\tEpochs: {self.epochs}")
        # print(f"\tDevice: {self.device}")
        # print(f"\tSeed: {self.seed}")
                
        self.projector = OWL2VecStarProjector(bidirectional_taxonomy=True)
        
        self.train_path = f'datasets/bin/{self.file_name}_train.owl'  
        self.valid_path = f'datasets/bin/{self.dataset_name}_val.owl'
        self.test_path = f'datasets/bin/{self.dataset_name}_test.owl'

        self._train_graph_path = f'datasets/bin/owl2vec/{self.file_name}/{self.file_name}_train.edgelist'
        self._valid_subsumption_graph_path = f'datasets/bin/owl2vec/{self.file_name}/{self.file_name}_val_subsumption.edgelist'
        self._test_subsumption_graph_path = f'datasets/bin/owl2vec/{self.file_name}/{self.file_name}_test_subsumption.edgelist'
        self._valid_membership_graph_path = f'datasets/bin/owl2vec/{self.file_name}/{self.file_name}_val_membership.edgelist'
        self._test_membership_graph_path = f'datasets/bin/owl2vec/{self.file_name}/{self.file_name}_test_membership.edgelist'

        self._classes_path = f'datasets/bin/owl2vec/{self.file_name}/{self.file_name}_classes.tsv'
        self._object_properties_path = f'datasets/bin/owl2vec/{self.file_name}/{self.file_name}_properties.tsv' 
        self._individuals_path = f'datasets/bin/owl2vec/{self.file_name}/{self.file_name}_individuals.tsv' 
        
        self.dataset = PathDataset(self.train_path, validation_path = self.valid_path, testing_path = self.test_path)
        
        self.model = KGEModule(kge_model,
                               triples_factory=self.triples_factory,
                               embedding_dim=self.emb_dim,
                               random_seed=self.seed)

    def _load_graph(self, path, mode="train"):
        logger.info(f"Generating Graph {path}...")
        if mode == "train":
            edges = self.projector.project(self.dataset.ontology)
        elif "valid" in mode:
            edges = self.projector.project(self.dataset.validation)
        elif "test" in mode:
            edges = self.projector.project(self.dataset.testing)

        edges = [(e.src, e.rel, e.dst) for e in edges]
        graph = pd.DataFrame(edges, columns=["head", "relation", "tail"])

        if mode == "train":
            graph.to_csv(path, sep="\t", header=None, index=False)
        elif "subsumption" in mode:
            graph = graph[graph["relation"] == "http://subclassof"]                 
            graph.to_csv(path, sep="\t", header=None, index=False)
        elif "membership" in mode:
            graph = graph[graph["relation"] == "http://type"]
            graph.to_csv(path, sep="\t", header=None, index=False)

        graph = pd.read_csv(path, sep="\t", header=None)
        graph.columns = ["head", "relation", "tail"]
                
        logger.info(f"Loaded {mode} graph with {len(graph)} edges")
        
        return graph
      
    @property
    def train_graph(self):
        if self._train_graph is not None:
            return self._train_graph

        self._train_graph = self._load_graph(self._train_graph_path, mode="train")
        return self._train_graph

    @property
    def valid_subsumption_graph(self):
        if self._valid_subsumption_graph is not None:
            return self._valid_subsumption_graph

        self._valid_subsumption_graph = self._load_graph(self._valid_subsumption_graph_path, mode="valid_subsumption")
        return self._valid_subsumption_graph
                                                            
    @property
    def valid_membership_graph(self):
        if self._valid_membership_graph is not None:
            return self._valid_membership_graph

        self._valid_membership_graph = self._load_graph(self._valid_membership_graph_path, mode="valid_membership")
        return self._valid_membership_graph

    @property
    def test_subsumption_graph(self):
        if self._test_subsumption_graph is not None:
            return self._test_subsumption_graph

        self._test_subsumption_graph = self._load_graph(self._test_subsumption_graph_path, mode="test_subsumption")
        return self._test_subsumption_graph

    @property
    def test_membership_graph(self):
        if self._test_membership_graph is not None:
            return self._test_membership_graph

        self._test_membership_graph = self._load_graph(self._test_membership_graph_path, mode="test_membership")
        return self._test_membership_graph
    
    @property
    def classes(self):
        if self._classes is not None:
            return self._classes

        logger.info(f"Generating Classes...")
        classes = set(self.dataset.ontology.getClassesInSignature())
        classes |= set(self.dataset.validation.getClassesInSignature())
        classes |= set(self.dataset.testing.getClassesInSignature())
        classes = sorted(list(classes))
        classes = [str(c.toStringID()) for c in classes]
        classes = pd.DataFrame(classes, columns=["class"])
        classes.to_csv(self._classes_path, sep="\t", header=None, index=False)
        classes = list(classes["class"].values.flatten())
        classes.sort()
        self._classes = classes
         
        return self._classes

    @property
    def object_properties(self):
        if self._object_properties is not None:
            return self._object_properties

        logger.info(f"Generating Properties...")
        properties = set(self.dataset.ontology.getObjectPropertiesInSignature())
        properties |= set(self.dataset.validation.getObjectPropertiesInSignature())
        properties |= set(self.dataset.testing.getObjectPropertiesInSignature())
        properties = sorted(list(properties))
        properties = [str(p.toStringID()) for p in properties]
        properties = pd.DataFrame(properties, columns=["property"])
        properties.to_csv(self._object_properties_path, sep="\t", header=None, index=False)
        properties = list(properties["property"].values.flatten())
        properties.sort()
        self._object_properties = properties
            
        return self._object_properties

    @property
    def individuals(self):
        if self._individuals is not None:
            return self._individuals

        logger.info(f"Generating Individuals...")
        individuals = set(self.dataset.ontology.getIndividualsInSignature())
        individuals |= set(self.dataset.validation.getIndividualsInSignature())
        individuals |= set(self.dataset.testing.getIndividualsInSignature())
        individuals = sorted(list(individuals))
        individuals = [str(i.toStringID()) for i in individuals]
        individuals = pd.DataFrame(individuals, columns=["individual"])
        individuals.to_csv(self._individuals_path, sep="\t", header=None, index=False)
        individuals = list(individuals["individual"].values.flatten())
        individuals.sort()
        self._individuals = individuals
        
        return self._individuals
    
    @property
    def model_path(self):
        if self._model_path is not None:
            return self._model_path

        self._model_path = f"models/owl2vec_{self.file_name}.model.pt"
        return self._model_path

    @property
    def node_to_id(self):
        if self._node_to_id is not None:
            return self._node_to_id

        graph_classes = set(self.train_graph["head"].unique()) | set(self.train_graph["tail"].unique())
        graph_classes |= set(self.valid_subsumption_graph["head"].unique()) | set(self.valid_subsumption_graph["tail"].unique())
        graph_classes |= set(self.valid_membership_graph["head"].unique()) | set(self.valid_membership_graph["tail"].unique())
        graph_classes |= set(self.test_subsumption_graph["head"].unique()) | set(self.test_subsumption_graph["tail"].unique())
        graph_classes |= set(self.test_membership_graph["head"].unique()) | set(self.test_membership_graph["tail"].unique())
        
        bot = "http://www.w3.org/2002/07/owl#Nothing"
        top = "http://www.w3.org/2002/07/owl#Thing"
        graph_classes.add(bot)
        graph_classes.add(top)
                
        ont_classes = set(self.classes)
        all_classes = list(graph_classes | ont_classes | set(self.individuals)) 
        all_classes.sort()
        self._node_to_id = {c: i for i, c in enumerate(all_classes)}
        logger.info(f"Number of graph nodes: {len(self._node_to_id)}")
        
        return self._node_to_id
    
    @property
    def id_to_node(self):
        if self._id_to_node is not None:
            return self._id_to_node
        
        id_to_node =  {v: k for k, v in self.node_to_id.items()}
        self._id_to_node = id_to_node
        return self._id_to_node
    
    @property
    def relation_to_id(self):
        if self._relation_to_id is not None:
            return self._relation_to_id

        graph_rels = list(self.train_graph["relation"].unique())
        graph_rels.sort()
        self._relation_to_id = {r: i for i, r in enumerate(graph_rels)}
        logger.info(f"Number of graph relations: {len(self._relation_to_id)}")
        return self._relation_to_id

    @property
    def id_to_relation(self):
        if self._id_to_relation is not None:
            return self._id_to_relation

        id_to_relation = {v: k for k, v in self.relation_to_id.items()}
        self._id_to_relation = id_to_relation
        return self._id_to_relation

    @property
    def triples_factory(self):
        if self._triples_factory is not None:
            return self._triples_factory

        tensor = []
        for row in self.train_graph.itertuples():
            tensor.append([self.node_to_id[row.head],
                           self.relation_to_id[row.relation],
                           self.node_to_id[row.tail]])

        tensor = th.LongTensor(tensor)
        self._triples_factory = TriplesFactory(tensor, self.node_to_id, self.relation_to_id, create_inverse_triples=True)
        return self._triples_factory

    @property
    def classes_ids(self):
        if self._classes_ids is not None:
            return self._classes_ids
        
        class_to_id = {c: self.node_to_id[c] for c in self.classes}
        ontology_classes_idxs = th.tensor(list(class_to_id.values()), dtype=th.long, device=self.device)
        self._classes_ids = ontology_classes_idxs
        return self._classes_ids

    @property
    def object_properties_ids(self):
        if self._object_property_ids is not None:
            return self._object_property_ids
        
        prop_to_id = {c: self.relation_to_id[c] for c in self.object_properties if c in self.relation_to_id}
        
        object_property_to_id = th.tensor(list(prop_to_id.values()), dtype=th.long, device=self.device)
        self._object_properties_ids = object_properties_ids
        return self._object_properties_ids

    @property
    def individuals_ids(self): 
        if self._individuals_ids is not None:
            return self._individuals_ids

        individual_to_id = {c: self.node_to_id[c] for c in self.individuals}
        individual_to_id = th.tensor(list(individual_to_id.values()), dtype=th.long, device=self.device)
        self._individuals_ids = individual_to_id
        return self._individuals_ids

    def create_graph_dataloader(self, mode="train", batch_size=None):        
        if mode == "train":
            graph = self.train_graph
        elif mode == "valid_subsumption":
            graph = self.valid_subsumption_graph
        elif mode == "valid_membership":
            graph = self.valid_membership_graph
        elif mode == "test_subsumption":
            graph = self.test_subsumption_graph
        elif mode == "test_membership":
            graph = self.test_membership_graph
        
        heads = [self.node_to_id[h] for h in graph["head"]]
        rels = [self.relation_to_id[r] for r in graph["relation"]]
        tails = [self.node_to_id[t] for t in graph["tail"]]

        heads = th.LongTensor(heads)
        rels = th.LongTensor(rels)
        tails = th.LongTensor(tails)
        
        dataloader = FastTensorDataLoader(heads, rels, tails,
                                          batch_size=batch_size, 
                                          shuffle=True)
        
        return dataloader
    
    def train(self):
        raise NotImplementedError
        
    def get_filtering_labels(self):
        logger.info("Getting predictions and labels")

        num_test_subsumption_heads = len(self.classes)
        num_test_subsumption_tails = len(self.classes)
        num_test_membership_heads = len(self.individuals)
        num_test_membership_tails = len(self.classes)

        subsumption_relation = "http://subclassof"
        membership_relation = "http://type"
            
        # get predictions and labels for test subsumption
        filter_subsumption_labels = np.ones((num_test_subsumption_heads, num_test_subsumption_tails), dtype=np.int32)
        filter_membership_labels = np.ones((num_test_membership_heads, num_test_membership_tails), dtype=np.int32)
        
        classes_ids = self.classes_ids.to(self.device)
        individuals_ids = self.individuals_ids.to(self.device)
        
        train_dataloader = self.create_graph_dataloader(mode="train", batch_size=self.batch_size)
        subsumption_triples = 0
        membership_triples = 0
        with th.no_grad():
            for head_idxs, rel_idxs, tail_idxs in train_dataloader:
                head_idxs = head_idxs.to(self.device)
                
                for i, head_graph_id in enumerate(head_idxs):
                    rel = rel_idxs[i]
                    rel_name = self.id_to_relation[rel.item()]
                    if rel_name == subsumption_relation:
                        tail_graph_id = tail_idxs[i]
                        
                        head_class = self.id_to_node[head_graph_id.item()]
                        tail_class = self.id_to_node[tail_graph_id.item()]
                        if not head_class in self.classes or not tail_class in self.classes:
                            continue
                                                
                        head_ont_id = th.where(self.classes_ids == head_graph_id)[0]
                        rel = rel_idxs[i]
                        tail_ont_id = th.where(self.classes_ids == tail_graph_id)[0]
                        
                        filter_subsumption_labels[head_ont_id, tail_ont_id] = 10000
                        subsumption_triples += 1
                    elif rel_name == membership_relation:
                        tail_graph_id = tail_idxs[i]

                        head_individual = self.id_to_node[head_graph_id.item()]
                        tail_class = self.id_to_node[tail_graph_id.item()]
                        if not head_individual in self.individuals or not tail_class in self.classes:
                            continue
                            
                        head_ont_id = th.where(self.individuals_ids == head_graph_id)[0]
                        tail_ont_id = th.where(self.classes_ids == tail_graph_id)[0]
                        
                        filter_membership_labels[head_ont_id, tail_ont_id] = 10000
                        membership_triples += 1
                    else:
                        continue

        logger.info(f"Subsumption filtering triples: {subsumption_triples}")
        logger.info(f"Membership filtering triples: {membership_triples}")
        return filter_subsumption_labels, filter_membership_labels

    def load_best_model(self):
        logger.info(f"Loading best model from {self.model_path}")
        self.model.load_state_dict(th.load(self.model_path))
        self.model = self.model.to(self.device)

    def compute_ranking_metrics(self, filtering_labels=None, mode="test_subsumption"):
        if "test" in mode:
            self.load_best_model()

        all_tail_ids = self.classes_ids.to(self.device)
        if "subsumption" in mode:
            all_head_ids = self.classes_ids.to(self.device)
        elif "membership" in mode:
            all_head_ids = self.individuals_ids.to(self.device)
        
        self.model.eval()
        mean_rank, filtered_mean_rank = 0, 0
        ranks, filtered_ranks = dict(), dict()
        rank_vals = []
        filtered_rank_vals = []
        if "test" in mode:
            mrr, filtered_mrr = 0, 0
            hits_at_1, fhits_at_1 = 0, 0
            hits_at_3, fhits_at_3 = 0, 0
            hits_at_10, fhits_at_10 = 0, 0
            hits_at_100, fhits_at_100 = 0, 0

        dataloader = self.create_graph_dataloader(mode, batch_size=self.test_batch_size)
        with th.no_grad():
            for head_idxs, rel_idxs, tail_idxs in dataloader:

                predictions = self.predict(head_idxs, rel_idxs, tail_idxs, mode)
                
                for i, graph_head in enumerate(head_idxs):
                    graph_tail = tail_idxs[i]
                    head = th.where(all_head_ids == graph_head)[0]
                    tail = th.where(all_tail_ids == graph_tail)[0]
                    
                    logger.debug(f"graph_tail: {graph_tail}")
                    
                    rel = rel_idxs[i]
                    preds = predictions[i]

                    orderings = th.argsort(preds, descending=True)
                    rank = th.where(orderings == tail)[0].item()
                    mean_rank += rank
                    rank_vals.append(rank)
                    if rank not in ranks:
                        ranks[rank] = 0
                    ranks[rank] += 1

                    if "test" in mode:
                        filt_labels = filtering_labels[head, :]
                        filt_labels[tail] = 1
                        filtered_preds = preds.cpu().numpy() * filt_labels
                        filtered_preds = th.from_numpy(filtered_preds).to(self.device)
                        filtered_orderings = th.argsort(filtered_preds, descending=True) 
                        filtered_rank = th.where(filtered_orderings == tail)[0].item()
                        filtered_mean_rank += filtered_rank
                        filtered_rank_vals.append(filtered_rank)
                        mrr += 1/(rank+1)
                        filtered_mrr += 1/(filtered_rank+1)
                    
                        if rank == 0:
                            hits_at_1 += 1
                        if rank < 3:
                            hits_at_3 += 1
                        if rank < 10:
                            hits_at_10 += 1
                        if rank < 100:
                            hits_at_100 += 1

                        if filtered_rank == 0:
                            fhits_at_1 += 1
                        if filtered_rank < 3:
                            fhits_at_3 += 1
                        if filtered_rank < 10:
                            fhits_at_10 += 1
                        if filtered_rank < 100:
                            fhits_at_100 += 1

                        if filtered_rank not in filtered_ranks:
                            filtered_ranks[filtered_rank] = 0
                        filtered_ranks[filtered_rank] += 1

            mean_rank /= dataloader.dataset_len
            if "test" in mode:
                mrr /= dataloader.dataset_len
                hits_at_1 /= dataloader.dataset_len
                hits_at_3 /= dataloader.dataset_len
                hits_at_10 /= dataloader.dataset_len
                hits_at_100 /= dataloader.dataset_len
                auc = self.compute_rank_roc(ranks)
                median_rank = np.median(rank_vals)
                percentile90 = np.percentile(rank_vals, 90)
                below_1000 = self.calculate_percentile_1000(ranks)
                
                filtered_mean_rank /= dataloader.dataset_len
                filtered_mrr /= dataloader.dataset_len
                fhits_at_1 /= dataloader.dataset_len
                fhits_at_3 /= dataloader.dataset_len
                fhits_at_10 /= dataloader.dataset_len
                fhits_at_100 /= dataloader.dataset_len
                fauc = self.compute_rank_roc(filtered_ranks)
                fmedian_rank = np.median(filtered_rank_vals)
                fpercentile90 = np.percentile(filtered_rank_vals, 90)
                fbelow_1000 = self.calculate_percentile_1000(filtered_ranks)
                
                raw_metrics = (mean_rank, mrr, median_rank, hits_at_1, hits_at_3, hits_at_10, hits_at_100, auc, percentile90, below_1000)
                filtered_metrics = (filtered_mean_rank, filtered_mrr, fmedian_rank, fhits_at_1, fhits_at_3, fhits_at_10, fhits_at_100, fauc, fpercentile90, fbelow_1000)
                print(f'MRR: {mrr:.3f}, Mean Rank: {mean_rank:.3f}, Median Rank: {median_rank:.3f}, Hits@1: {hits_at_1:.3f}, Hits@3: {hits_at_3:.3f}, Hits@10: {hits_at_10:.3f}, Hits@100: {hits_at_100:.3f}')
        if "test" in mode:
            return raw_metrics, filtered_metrics
        else:
            return mean_rank
                                                                            
    def normal_forward(self, head_idxs, rel_idxs, tail_idxs, n_classes):
        logits = self.model.predict((head_idxs, rel_idxs, tail_idxs))
        logger.debug(f"logits shape before reshape: {logits.shape}")
        logits = logits.reshape(-1, n_classes)
        logger.debug(f"logits shape after reshape: {logits.shape}")
        return logits

    def predict(self, heads, rels, tails, mode):
        aux = heads.to(self.device)
        num_heads = len(heads)
        tail_ids = self.classes_ids.to(self.device)                              
        heads = heads.to(self.device)
        heads = heads.repeat(len(tail_ids), 1).T
        heads = heads.reshape(-1)
        rels = rels.to(self.device)
        rels = rels.repeat(len(tail_ids),1).T
        rels = rels.reshape(-1)
        eval_tails = tail_ids.repeat(num_heads)
        logits = self.normal_forward(heads, rels, eval_tails, len(tail_ids))
        return logits
        
    def test(self):
        logger.info("Testing ontology completion...")
        filtering_labels = self.get_filtering_labels()
        print('Membership:')
        membership_metrics = self.compute_ranking_metrics(filtering_labels[1], "test_membership")       
        print('Subsumption:')
        subsumption_metrics = self.compute_ranking_metrics(filtering_labels[0], "test_subsumption")
        return subsumption_metrics, membership_metrics

    def compute_rank_roc(self, ranks):
        n_tails = len(self.classes_ids)   
        auc_x = list(ranks.keys())
        auc_x.sort()
        auc_y = []
        tpr = 0
        sum_rank = sum(ranks.values())
        for x in auc_x:
            tpr += ranks[x]
            auc_y.append(tpr / sum_rank)
        auc_x.append(n_tails)
        auc_y.append(1)
        auc = np.trapz(auc_y, auc_x) / n_tails
        return auc
            
    def save_embeddings_data(self):
        out_class_file = f"datasets/bin/owl2vec/{self.file_name}/{self.file_name}_class_embeddings.pkl"
        out_individual_file = f"datasets/bin/owl2vec/{self.file_name}/{self.file_name}_individual_embeddings.pkl"
        out_role_file = f"datasets/bin/owl2vec/{self.file_name}/{self.file_name}_role_embeddings.pkl"
        out_triples_factory_file = f"datasets/bin/owl2vec/{self.file_name}/{self.file_name}_triples_factory.pkl"
        
        cls_ids = [self.node_to_id[n] for n in self.classes]
        ind_ids = [self.node_to_id[n] for n in self.individuals]
        role_ids = [self.relation_to_id[n] for n in ["http://subclassof", "http://type"]]
            
        cls_df = pd.DataFrame(list(zip(self.classes, cls_ids)), columns=["class", "node_id"])
        inds_df = pd.DataFrame(list(zip(self.individuals, ind_ids)) , columns=["individual", "node_id"])
        role_df = pd.DataFrame(list(zip(["http://subclassof", "http://type"], role_ids)), columns=["role", "relation_id"])
        
        cls_df.to_pickle(out_class_file)
        logger.info(f"Saved class data to {out_class_file}")
        inds_df.to_pickle(out_individual_file)
        logger.info(f"Saved individual data to {out_individual_file}")
        role_df.to_pickle(out_role_file)
        logger.info(f"Saved role data to {out_role_file}")
        
        pkl.dump(self.triples_factory, open(out_triples_factory_file, "wb"))
        logger.info(f"Saved triples factory to {out_triples_factory_file}")

    def calculate_percentile_1000(self, scores):
        ranks_1000 = []
        for item in scores:
            if item < 1000:
                ranks_1000.append(item)
        n_1000 = len(ranks_1000)
        nt = len(scores)
        percentile = (n_1000/nt)*100
        return percentile