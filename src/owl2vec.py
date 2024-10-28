import torch.optim as optim
import torch 
import torch.nn as nn
from .graph_model import GraphModel

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
                print(f"Epoch: {epoch}, Loss: {graph_loss:.3f}")

        self.save_embeddings_data()