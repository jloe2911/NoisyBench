import torch.optim as optim
import torch 
import torch.nn as nn
from tqdm import tqdm, trange
import logging
from .graph_model import GraphModel

class OWL2Vec(GraphModel):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
               
    def train(self):                                                                                  
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        min_lr = self.lr/10
        max_lr = self.lr
        
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                                      base_lr=min_lr,
                                                      max_lr=max_lr, 
                                                      step_size_up = 20,
                                                      cycle_momentum = False)

        criterion_bpr = nn.LogSigmoid()
        
        self.model = self.model.to(self.device)

        train_graph_dataloader = self.create_graph_dataloader(mode="train", batch_size=self.batch_size)

        tolerance = 0
        best_loss = float("inf")
        best_mr = float("inf")
        classes_ids = torch.tensor(list(self.classes_ids), dtype=torch.long, device=self.device)

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
                for i in range(self.num_negs):
                    neg_tail = torch.randint(0, len(self.node_to_id), (len(head),), device=self.device)
                    data = (head, rel, neg_tail)
                    neg_logits += self.model.forward(data)
                                                            
                neg_logits /= (self.num_negs)

                batch_loss = -criterion_bpr(pos_logits - neg_logits - self.margin).mean()
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                graph_loss += batch_loss.item()

            graph_loss /= len(train_graph_dataloader)
            
            valid_subsumption_mr = self.compute_ranking_metrics(mode="valid_subsumption")
            valid_membership_mr = self.compute_ranking_metrics(mode="valid_membership")
            total_mr = (valid_subsumption_mr + valid_membership_mr)
            sub_weight = 0.5 
            mem_weight = 0.5 
            valid_mr = sub_weight*valid_subsumption_mr + mem_weight*valid_membership_mr
            
            if valid_mr < best_mr:
                best_mr = valid_mr
                torch.save(self.model.state_dict(), self.model_path)
                tolerance = self.initial_tolerance+1
            if epoch % 25 == 0:
                print(f"Epoch: {epoch}, Training loss: {graph_loss:.3f}, Validation mean ranks: sub-{valid_subsumption_mr:.3f}, mem-{valid_membership_mr:.3f}, avg-{valid_mr:.3f}")

            tolerance -= 1
            if tolerance == 0:
                print("Early stopping")
                break

        self.save_embeddings_data()        