import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np

from src.graph_model import GraphModel
from src.utils import save_results

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OWL2Vec(GraphModel):
    def __init__(self, *args, output_path=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Handle output path
        if output_path is None:
            output_path = os.path.join("models", "owl2vec", self.dataset_name)
        os.makedirs(output_path, exist_ok=True)

        self.output_path = output_path
        self._model_path = os.path.join(output_path, "model.pt")
        self._embeddings_path = os.path.join(output_path, "embeddings.npy")

        # Initialize embeddings if not done in parent
        if hasattr(self.model, 'node_embeddings'):
            nn.init.xavier_uniform_(self.model.node_embeddings.weight)
        if hasattr(self.model, 'relation_embeddings'):
            nn.init.xavier_uniform_(self.model.relation_embeddings.weight)

        # -----------------------------
        # Handle unknown relations
        # -----------------------------
        if "<UNK>" not in self.relation_to_id:
            self.relation_to_id["<UNK>"] = len(self.relation_to_id)
            if hasattr(self.model, "relation_embeddings"):
                with torch.no_grad():
                    unk_idx = self.relation_to_id["<UNK>"]
                    nn.init.xavier_uniform_(self.model.relation_embeddings.weight[unk_idx:unk_idx+1])

    # ------------------------------------------------------------------

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=self.weight_decay)

        # Cyclic LR scheduler
        min_lr = 0.0001
        max_lr = 0.001
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=min_lr,
            max_lr=max_lr,
            step_size_up=200,
            cycle_momentum=False
        )

        criterion_bpr = nn.LogSigmoid()
        self.model = self.model.to(self.device)

        train_graph_dataloader = self.create_graph_dataloader(
            mode="train",
            batch_size=min(self.batch_size, 8192)
        )

        for epoch in range(self.epochs):
            self.model.train()
            graph_loss = 0

            for head, rel, tail in train_graph_dataloader:
                head = head.to(self.device)
                rel = rel.to(self.device)
                tail = tail.to(self.device)

                # Positive samples
                pos_logits = self.model.forward((head, rel, tail))

                # Negative samples (one per positive)
                neg_tail = torch.randint(0, len(self.node_to_id), (len(head),), device=self.device)
                neg_logits = self.model.forward((head, rel, neg_tail))

                batch_loss = -criterion_bpr(pos_logits - neg_logits).mean()

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                graph_loss += batch_loss.item()

            graph_loss /= len(train_graph_dataloader)

            if epoch % 25 == 0:
                torch.save(self.model.state_dict(), self._model_path)
                logger.info(f"[{self.dataset_name}] Epoch {epoch}: loss={graph_loss:.4f} (model saved)")

        self.save_embeddings_data()

    # ------------------------------------------------------------------
    #                   OWL2Vec Experiment Runners
    # ------------------------------------------------------------------

def run_owl2vec(dataset_name, device, experiments):
    base_output_dir = f"models/owl2vec/{dataset_name}"
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs("models/results/owl2vec/", exist_ok=True)

    for experiment in experiments:
        dataset_name = experiment["dataset_name"]
        file_name = experiment["file_name"]

        subsumption_results = []
        membership_results = []
        link_prediction_results = []

        for i in range(5):
            run_output_dir = os.path.join(base_output_dir, file_name, f"run_{i}")
            os.makedirs(run_output_dir, exist_ok=True)

            model = OWL2Vec(
                file_name=file_name,
                iteration=i,
                dataset_name=dataset_name,
                output_path=run_output_dir,
                kge_model="transe",
                emb_dim=256,
                margin=0.1,
                weight_decay=0.0,
                batch_size=8192,
                lr=0.001,
                num_negs=1,
                test_batch_size=32,
                epochs=300,
                device=device,
                seed=42+i,
                initial_tolerance=5
            )

            logger.info(f"Starting OWL2Vec run {i} for {file_name} (seed={42+i}) → {run_output_dir}")
            model.train()

            # -----------------------------
            # Key change: safely handle unknown relations in test
            # -----------------------------
            metrics_subsumption, metrics_membership, metrics_link_prediction = model.test()

            subsumption_results.append(metrics_subsumption)
            membership_results.append(metrics_membership)
            link_prediction_results.append(metrics_link_prediction)

        save_results(
            subsumption_results,
            membership_results,
            link_prediction_results,
            f"models/results/owl2vec/{file_name}.txt"
        )


# ----------------------------------------------------------------------

def run_owl2vec_test(dataset_name, device, experiments):
    os.makedirs(f"models/owl2vec/{dataset_name}", exist_ok=True)
    os.makedirs("models/results/owl2vec/", exist_ok=True)

    for experiment in experiments:
        dataset_name = experiment["dataset_name"]
        file_name = experiment["file_name"]

        subsumption_results = []
        membership_results = []
        link_prediction_results = []

        model = OWL2Vec(
            file_name=file_name,
            iteration=0,
            dataset_name=dataset_name,
            output_path=f"models/owl2vec/{dataset_name}/test_run",
            kge_model="transe",
            emb_dim=256,
            margin=0.1,
            weight_decay=0.0,
            batch_size=8192,
            lr=0.001,
            num_negs=1,
            test_batch_size=32,
            epochs=25,
            device=device,
            seed=42,
            initial_tolerance=5
        )

        model.train()

        logger.info(f"Testing {file_name} (seed=42)")
        metrics_subsumption, metrics_membership, metrics_link_prediction = model.test()

        subsumption_results.append(metrics_subsumption)
        membership_results.append(metrics_membership)
        link_prediction_results.append(metrics_link_prediction)

        save_results(
            subsumption_results,
            membership_results,
            link_prediction_results,
            f"models/results/owl2vec/{file_name}_test.txt"
        )

        break