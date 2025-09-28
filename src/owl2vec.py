import os 

import torch.optim as optim
import torch 
import torch.nn as nn

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from src.graph_model import GraphModel
from src.utils import save_results

class OWL2Vec(GraphModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
               
    def train(self):                                                                                  
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        min_lr = self.lr / 10
        max_lr = self.lr
        
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                                      base_lr=min_lr,
                                                      max_lr=max_lr, 
                                                      step_size_up=20,
                                                      cycle_momentum=False)

        criterion_bpr = nn.LogSigmoid()
        
        self.model = self.model.to(self.device)
        train_graph_dataloader = self.create_graph_dataloader(mode="train", batch_size=self.batch_size)

        for epoch in range(self.epochs):
            self.model.train()
            graph_loss = 0
            
            for head, rel, tail in train_graph_dataloader:
                head = head.to(self.device)
                rel = rel.to(self.device)
                tail = tail.to(self.device)
                
                data = (head, rel, tail)
                pos_logits = self.model.forward(data)

                neg_logits = 0
                for _ in range(self.num_negs):
                    neg_tail = torch.randint(0, len(self.node_to_id), (len(head),), device=self.device)
                    data = (head, rel, neg_tail)
                    neg_logits += self.model.forward(data)
                                                            
                neg_logits /= self.num_negs
                batch_loss = -criterion_bpr(pos_logits - neg_logits - self.margin).mean()
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                graph_loss += batch_loss.item()

            graph_loss /= len(train_graph_dataloader)
            
            if epoch % 25 == 0:
                torch.save(self.model.state_dict(), self.model_path)

        self.save_embeddings_data()

def run_owl2vec(dataset_name, device, experiments):
    os.makedirs(f'datasets/bin/owl2vec/{dataset_name}', exist_ok=True)
    os.makedirs(f'models/results/owl2vec/', exist_ok=True)

    for experiment in experiments:
        dataset_name = experiment['dataset_name']
        file_name = experiment['file_name']
        
        subsumption_results = []
        membership_results = []
        link_prediction_results = []
        
        for i in range(5):
            model = OWL2Vec(
                file_name=file_name,
                iteration = i, 
                dataset_name=dataset_name,
                kge_model='transe',
                emb_dim=256,
                margin=0.1,
                weight_decay=0.0,
                batch_size=4096*8,
                lr=0.0001,
                num_negs=4,
                test_batch_size=32,
                epochs=500,
                device=device,
                seed=42,
                initial_tolerance=5
            )
            
            model.train()
            
            logging.info(f'{file_name}:')
            metrics_subsumption, metrics_membership, metrics_link_prediction = model.test()

            subsumption_results.append(metrics_subsumption)
            membership_results.append(metrics_membership)
            link_prediction_results.append(metrics_link_prediction)

        save_results(subsumption_results, membership_results, link_prediction_results, f'models/results/owl2vec/{file_name}.txt')

def run_owl2vec_test(dataset_name, device, experiments):
    os.makedirs(f'datasets/bin/owl2vec/{dataset_name}', exist_ok=True)
    os.makedirs(f'models/results/owl2vec/', exist_ok=True)

    for experiment in experiments:
        dataset_name = experiment['dataset_name']
        file_name = experiment['file_name']
        
        subsumption_results = []
        membership_results = []
        link_prediction_results = []
        
        model = OWL2Vec(
            file_name=file_name,
            iteration = 0, 
            dataset_name=dataset_name,
            kge_model='transe',
            emb_dim=256,
            margin=0.1,
            weight_decay=0.0,
            batch_size=4096*8,
            lr=0.0001,
            num_negs=4,
            test_batch_size=32,
            epochs=500,
            device=device,
            seed=42,
            initial_tolerance=5
        )
        
        model.train()
        
        logging.info(f'{file_name}:')
        metrics_subsumption, metrics_membership, metrics_link_prediction = model.test()

        subsumption_results.append(metrics_subsumption)
        membership_results.append(metrics_membership)
        link_prediction_results.append(metrics_link_prediction)

        save_results(subsumption_results, membership_results, link_prediction_results, f'models/results/owl2vec/{file_name}_test.txt')

        break