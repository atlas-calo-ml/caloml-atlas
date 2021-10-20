# Stuff for saving/loading models.
from qkeras.utils import model_save_quantized_weights
from tensorflow.keras.optimizers import Adam


# Given a quantized model (of type tensorflow.python.keras.engine.functional.Functional),
# save the quantized weights. (This is just a wrapper function).
def SaveQuantizedWeights(qmodel, qmodelfile):
    model_save_quantized_weights(qmodel, qmodelfile)
    return

# Model compilation: If we're saving models without compilation,
# which seems to avoid some bugs/weird behaviour from AutoQKeras-trained models,
# we will need to recompile in order to do things like more training or "evaluation". (Not prediction.)
# This requires compiling the model, for which we'll default to the Adam optimizer.
def GetCompiledModel(qmodel, lr, loss='categorical_crossentropy', metrics = ['acc']):
    optimizer = Adam(learning_rate=lr)
    qmodel.compile(loss=loss, optimizer=optimizer, metrics = metrics)
    return qmodel