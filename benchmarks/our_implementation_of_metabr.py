"""Per-learner competence map, in the spirit of Meta-BR.

A meta-learner predicts per instance whether each member is correct, and the
vote is weighted by that estimated accuracy alone. Each member is modelled
independently through binary relevance, so dependencies between members are not
represented; this is the first-order competence map that LENS replaces with a
high-order one. Unlabelled instances are discarded.

Reference: Gama and Kosina, Recurrent concepts in data streams classification,
Knowledge and Information Systems 40, 2014.
"""
from _runner import Method, main
from lens.config import CONFIG_2, INF_MLHAT_A, TR_NONE

METHOD = Method(
    key=CONFIG_2,
    name="Meta-BR",
    reference="Gama and Kosina, Knowledge and Information Systems 40, 2014",
    inference_mode=INF_MLHAT_A,
    training_mode=TR_NONE,
)

if __name__ == "__main__":
    main(METHOD)
