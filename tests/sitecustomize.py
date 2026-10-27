# Se carga automáticamente en cada `python` que lanza tests/run_pipeline.sh
# (porque ese script agrega tests/ al PYTHONPATH). No afecta al uso normal del repo.
#
# Por qué existe: en CPU, la librería oneDNN de PyTorch (torch.backends.mkldnn)
# calcula mal, de forma aleatoria, los gradientes de ChannelNet al entrenar.
# Resultado: grad_norm = inf/nan, la etapa 1 no aprende, el filtro de la etapa 3
# queda con 0 ejemplos y el pipeline se cae. Comprobado con torch 2.4.1 y 2.14.1.
# Sin oneDNN, los gradientes son correctos y deterministas.
# En GPU se usa cuDNN, no oneDNN, así que el entrenamiento real no se ve afectado.
#
# - Con gradientes activos (entrenamiento): oneDNN APAGADO -> gradientes correctos.
# - Sin gradientes (evaluación, filtro, inferencia): oneDNN ENCENDIDO. Ahí calcula
#   bien, y sin él la convolución espacial de ChannelNet necesita unos 8 GB de RAM.
try:
    import torch

    torch.backends.mkldnn.enabled = False
    _call = torch.nn.Module.__call__

    def _call_mkldnn_sin_gradientes(self, *args, **kwargs):
        if torch.is_grad_enabled():
            return _call(self, *args, **kwargs)
        with torch.backends.mkldnn.flags(enabled=True):
            return _call(self, *args, **kwargs)

    torch.nn.Module.__call__ = _call_mkldnn_sin_gradientes
except Exception:
    pass
