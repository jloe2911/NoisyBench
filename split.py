import os
import random
import click as ck
from owlready2 import get_ontology, sync_reasoner_pellet
from rdflib import Graph, RDFS, RDF, OWL, Literal, BNode

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from src.utils import get_namespace


# =============================
# CLICK PARAMETERS
# =============================
@ck.command()
@ck.option(
    "--dataset_name",
    type=ck.Choice(['pizza', 'pizza_100', 'pizza_250', 'family', 'OWL2DL-1_100']),
    required=True,
    help="Dataset ontology to process.",
)
@ck.option(
    "--tbox",
    type=ck.Choice(['pizza_TBOX', 'family_TBOX', 'OWL2DL-1_TBOX']),
    required=True,
    help="Only TBOX to process."
)
def main(dataset_name, tbox):

    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"TBOX: {tbox}")

    # =============================
    # FULL PIPELINE
    # =============================
    def get_inferences(dataset_name):
        onto = get_ontology(f"ontologies/{dataset_name}.owl").load()
        with onto:
            sync_reasoner_pellet(infer_property_values=True)
        logger.info("Reasoning complete.")
        onto.save(f"ontologies/{dataset_name}_inferred.owl", format="rdfxml")

    def get_inferences_filtered(dataset_name):
        g_orig = Graph()
        g_orig.parse(f"ontologies/{dataset_name}.owl", format="xml")
        logger.info("Original:", len(g_orig))

        g_inf = Graph()
        g_inf.parse(f"ontologies/{dataset_name}_inferred.owl", format="xml")
        logger.info("Inferred:", len(g_inf))

        allowed_pred_prefix = get_namespace(dataset_name)
        allowed_predicates = {RDFS.subClassOf, RDF.type}

        g_orig_set = set(g_orig)
        g_filtered = Graph()

        for s, p, o in g_inf:
            if (
                (str(p).startswith(allowed_pred_prefix) or p in allowed_predicates)
                and not isinstance(o, Literal)
                and not isinstance(s, BNode)
                and not isinstance(o, BNode)
                and s not in {OWL.Thing, OWL.Nothing}
                and o not in {OWL.Thing, OWL.Nothing}
                and o != RDFS.Resource
                and (s, p, o) not in g_orig_set
            ):
                g_filtered.add((s, p, o))

        return g_filtered

    def get_membership_and_lp(g_filtered, namespace):
        ns_str = str(namespace)
        membership = list(g_filtered.triples((None, RDF.type, None)))
        lp = [(s, p, o) for (s, p, o) in g_filtered if str(p).startswith(ns_str)]
        logger.info("Membership:", len(membership))
        logger.info("LP:", len(lp))
        return membership, lp

    def split(triples):
        triples = list(triples)
        random.shuffle(triples)
        n = len(triples)
        a = int(0.7 * n)
        b = a + int(0.15 * n)
        return triples[:a], triples[a:b], triples[b:]

    def triples_to_graph(triples, base_graph=None):
        G = Graph()
        if base_graph:
            G += base_graph
        for t in triples:
            G.add(t)
        return G

    # Run pipeline
    g = Graph()
    g.parse(f"ontologies/{dataset_name}.owl", format="xml")

    g_tbox = Graph()
    g_tbox.parse(f"ontologies/{tbox}.owl", format="xml")

    # get_inferences(dataset_name) -> or use PROTEGE
    g_filtered = get_inferences_filtered(dataset_name)

    NS = get_namespace(dataset_name)
    membership, lp = get_membership_and_lp(g_filtered, NS)

    train_m, val_m, test_m = split(membership)
    train_lp, val_lp, test_lp = split(lp)

    G_train = triples_to_graph(train_m + train_lp, base_graph=g)
    G_val = triples_to_graph(val_m + val_lp, base_graph=g_tbox)
    G_test = triples_to_graph(test_m + test_lp, base_graph=g_tbox)

    os.makedirs("datasets", exist_ok=True)
    G_train.serialize(f"datasets/{dataset_name}_train.owl", format="xml")
    G_val.serialize(f"datasets/{dataset_name}_val.owl", format="xml")
    G_test.serialize(f"datasets/{dataset_name}_test.owl", format="xml")

    logger.info("Done.")


if __name__ == "__main__":
    main()