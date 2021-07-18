# Pruning isn't specifically qkeras-related, but it's most useful in the context
# of making small models.

from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning

def GetPrunedModel(model, target_sparsity=0.75, begin_step=2000, frequency=100):
    pruning_params = {"pruning_schedule" : pruning_schedule.ConstantSparsity(target_sparsity, begin_step=begin_step, frequency=frequency)}
    pmodel = prune.prune_low_magnitude(model, **pruning_params)
    return pmodel