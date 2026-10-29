"""[EPN] Compatibilidad entre versiones de transformers, compartida por todas las etapas.

La función crear_training_arguments() la escribió el equipo para la etapa 1
(train_eeg_classifier.py); se movió aquí para que finetune_llm.py (etapas 2 y 3)
use exactamente la misma lógica.

Con la versión fijada en requirements.txt (transformers 4.44.0) no cambia nada.
Con transformers 5.x adapta los argumentos que cambiaron de nombre o desaparecieron.
"""
import inspect
import logging

from transformers import TrainingArguments

try:
    from transformers.training_args import OptimizerNames
except ImportError:
    try:
        from transformers import OptimizerNames
    except ImportError:
        OptimizerNames = None

logger = logging.getLogger(__name__)


def crear_training_arguments(**kwargs):
    """[EPN] TrainingArguments con los mismos valores del original en cualquier versión.

    Si la versión instalada no admite algún argumento, solo ese se adapta o se omite y
    queda un aviso en el registro, en lugar de detener el entrenamiento.
    """
    aceptados = set(inspect.signature(TrainingArguments.__init__).parameters)

    # eval_strategy se llamaba evaluation_strategy antes de transformers 4.41.
    if "eval_strategy" in kwargs and "eval_strategy" not in aceptados and "evaluation_strategy" in aceptados:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    elif "evaluation_strategy" in kwargs and "evaluation_strategy" not in aceptados and "eval_strategy" in aceptados:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

    # transformers 5 eliminó warmup_ratio: ahora warmup_steps acepta una fracción (< 1).
    # Se convierte para conservar el calentamiento del artículo en vez de perderlo.
    if "warmup_ratio" in kwargs and "warmup_ratio" not in aceptados:
        ratio = kwargs.pop("warmup_ratio")
        if ratio:
            kwargs["warmup_steps"] = float(ratio)
            logger.info("warmup_ratio=%s se pasa como warmup_steps=%s (transformers 5).", ratio, ratio)

    # adamw_hf (el AdamW propio de transformers) no existe en versiones nuevas;
    # adamw_torch es el AdamW de PyTorch con los mismos hiperparámetros.
    if OptimizerNames is not None:
        optimizadores = {opcion.value for opcion in OptimizerNames}
    else:
        optimizadores = set()

    if kwargs.get("optim") and optimizadores and kwargs["optim"] not in optimizadores:
        logger.warning("El optimizador %s no existe en esta versión; se usa adamw_torch.", kwargs["optim"])
        kwargs["optim"] = "adamw_torch"

    for nombre in [n for n in list(kwargs.keys()) if n not in aceptados]:
        logger.warning("TrainingArguments no admite '%s' en esta versión; se omite.", nombre)
        kwargs.pop(nombre)
    return TrainingArguments(**kwargs)
