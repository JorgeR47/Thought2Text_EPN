"""
Etapa 1 de Thought2Text: entrenamiento del codificador EEG (ChannelNet).

El codificador recibe la señal EEG y aprende dos cosas a la vez:
  - clasificar la señal en una de las clases de objeto (entropía cruzada), y
  - acercar su vector de 512 dimensiones al embedding CLIP de la imagen vista (MSE).

Versión revisada por el equipo EPN para replicar el entrenamiento del artículo.
Los cambios respecto al original están marcados con "[EPN]" y explicados en el informe.
No cambian el método ni los hiperparámetros del artículo.
"""
import inspect
from collections.abc import Mapping
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    CLIPVisionModelWithProjection,
    Trainer,
    TrainingArguments,
)
try:
    from transformers.training_args import OptimizerNames
except ImportError:
    try:
        from transformers import OptimizerNames
    except ImportError:
        OptimizerNames = None

from args import get_args_for_encoder_training
from channelnet.config import EEGModelConfig
from channelnet.model import ChannelNetModel
from datautils import EEGDataset, Splitter
from loss import MSELoss

# [EPN] Se eliminó `import eval`: no existe ese módulo. Python importaba la carpeta eval/
# del repositorio como paquete vacío y, de paso, tapaba la función eval() de Python.
# [EPN] Se eliminó `import evaluate`: la exactitud se calcula directamente con exactitud(),
# eliminando dependencias externas (scikit-learn) y descargas innecesarias de Hugging Face.

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_seed(seed):
    """Set seed for reproducibility"""
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for numpy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # disable to ensure reproducibility


def exactitud(predicciones, referencias):
    """[EPN] Exactitud (accuracy): aciertos / total.

    Es el mismo cálculo numérico que evaluate.load("accuracy"), pero sin descargar la métrica de
    Internet ni requerir scikit-learn, garantizando ejecución reproducible y offline.
    """
    if not referencias:
        return 0.0
    aciertos = sum(int(p == r) for p, r in zip(predicciones, referencias))
    return float(aciertos) / len(referencias)


def crear_training_arguments(**kwargs):
    """[EPN] TrainingArguments con los mismos valores del original en cualquier versión.

    Con la versión fijada en requirements.txt (transformers 4.44.0) no cambia nada.
    Si la versión instalada no admite algún argumento, solo ese se adapta o se omite y
    queda un aviso en el registro, en lugar de detener el entrenamiento.
    """
    aceptados = set(inspect.signature(TrainingArguments.__init__).parameters)

    # eval_strategy se llamaba evaluation_strategy antes de transformers 4.41.
    if "eval_strategy" in kwargs and "eval_strategy" not in aceptados and "evaluation_strategy" in aceptados:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    elif "evaluation_strategy" in kwargs and "evaluation_strategy" not in aceptados and "eval_strategy" in aceptados:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")

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


class EEGEncoderTrainer(Trainer):
    def __init__(
        self,
        emb_loss_fn=None,
        cls_loss_fn=None,
        clip_model=None,
        data_loaders=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.emb_loss_fn = emb_loss_fn
        self.cls_loss_fn = cls_loss_fn
        self.clip_model = clip_model
        self.data_loaders = data_loaders
        self.softmax = torch.nn.Softmax(dim=1)
        if hasattr(self, "args") and hasattr(self.args, "device"):
            self.device = self.args.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # [EPN] **kwargs: desde transformers 4.46, Trainer llama a compute_loss con un
    # argumento extra (num_items_in_batch). Con 4.44 no llega nada y funciona igual.
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self.model.train()
        img_data, eeg, labels = inputs

        # [EPN] Manejo robusto de datos de imagen ya sea BatchFeature/dict o tensor directo
        pixel_values = (
            img_data["pixel_values"]
            if isinstance(img_data, Mapping)
            else img_data
        )
        pixel_values = pixel_values.to(self.clip_model.device)

        # [EPN] CLIP está congelado: sin gradientes no se guarda su grafo en memoria,
        # previniendo fugas de memoria VRAM (OOM).
        with torch.no_grad():
            image_embeddings = self.clip_model(
                pixel_values=pixel_values
            ).image_embeds

        emb_output, cls_output = model(eeg)
        image_embeddings = image_embeddings.to(emb_output.device)
        labels = labels.to(cls_output.device)

        emb_loss = self.emb_loss_fn(E1=emb_output, E2=image_embeddings)
        cls_loss = self.cls_loss_fn(cls_output, labels)
        loss = cls_loss + emb_loss
        self.device = eeg.device
        return (loss, cls_output) if return_outputs else loss

    def get_train_dataloader(self):
        return self.data_loaders["train"]

    def get_eval_dataloader(self, eval_dataset=None):
        return self.data_loaders["val"]

    def get_test_dataloader(self, test_dataset: Dataset = None) -> DataLoader:
        return self.data_loaders["test"]

    def _evaluar(self, dataloader):
        """[EPN] Pérdida total y exactitud sobre un DataLoader, sin gradientes.

        Reemplaza los dos bucles casi idénticos del original (validación y prueba),
        corrigiendo el error de atributo donde un diccionario no admite .to(device).
        """
        total_loss = 0.0
        all_labels = []
        all_preds = []
        device = (
            next(self.model.parameters()).device
            if list(self.model.parameters())
            else (self.args.device if hasattr(self, "args") else self.device)
        )

        with torch.no_grad():
            for image_raw, eeg_data, labels in tqdm(dataloader):
                pixel_values = (
                    image_raw["pixel_values"]
                    if isinstance(image_raw, dict)
                    else image_raw
                )
                pixel_values = pixel_values.to(self.clip_model.device)
                eeg_data = eeg_data.to(device)
                labels = labels.to(device)

                image_embeddings = self.clip_model(
                    pixel_values=pixel_values
                ).image_embeds
                image_embeddings = image_embeddings.to(device)

                emb_output, cls_output = self.model(eeg_data)
                emb_loss = self.emb_loss_fn(E1=emb_output, E2=image_embeddings)
                cls_loss = self.cls_loss_fn(cls_output, labels)
                total_loss += (cls_loss + emb_loss).item()

                preds = self.softmax(cls_output).argmax(dim=1)
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

        return total_loss, exactitud(all_preds, all_labels)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        self.model.eval()
        eval_loss, eval_acc = self._evaluar(self.get_eval_dataloader(eval_dataset=None))
        print({"eval_loss": eval_loss, "acc": eval_acc})

        # Prueba informativa en cada época. El mejor modelo se elige solo por validación.
        test_loss, test_acc = self._evaluar(self.get_test_dataloader(test_dataset=None))
        print({"test_loss": test_loss, "acc": test_acc})

        # Igual que el original: Trainer conserva el checkpoint con menor "eval_loss",
        # así que devolver -exactitud hace que elija el de mayor exactitud de validación.
        # [EPN] Se añaden las exactitudes para verlas también en TensorBoard.
        metricas = {
            "eval_loss": -eval_acc,
            "eval_accuracy": eval_acc,
            "test_accuracy": test_acc,
        }
        self.log(metricas)
        self.model.train()
        return metricas


def set_gradients(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def main():
    args = get_args_for_encoder_training()
    set_seed(42)
    clip_model = CLIPVisionModelWithProjection.from_pretrained(args.clip_model)
    clip_model.to(args.device)
    clip_model.requires_grad_(False)
    set_gradients(clip_model, False)
    clip_model.eval()

    dataset = EEGDataset(args=args)
    loaders = {
        split: DataLoader(
            Splitter(
                dataset,
                split_path=args.splits_path,
                split_num=args.split_num,
                split_name=split,
            ),
            batch_size=args.batch_size,
            # [EPN] Entrenamiento igual que el original (mezclado y sin el último lote
            # incompleto). Validación y prueba: todos los ejemplos y en orden fijo, para
            # que la exactitud no cambie por azar ni se descarten muestras.
            drop_last=(split == "train"),
            shuffle=(split == "train"),
        )
        for split in ["train", "val", "test"]
    }

    config = EEGModelConfig()

    config.save_pretrained(args.output)
    model = ChannelNetModel(config=config)

    training_arguments = crear_training_arguments(
        output_dir=args.output,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        optim=args.optim,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        # [EPN] Restaurados: forman parte de la configuración del artículo.
        warmup_ratio=args.warmup_ratio,
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        load_best_model_at_end=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        # [EPN] Sin esto, como wandb está en requirements.txt, Trainer intenta usarlo y
        # pide iniciar sesión. Las etapas 2 y 3 ya usan tensorboard.
        report_to="tensorboard",
    )
    trainer = EEGEncoderTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        eval_dataset=dataset,
        emb_loss_fn=MSELoss(),
        cls_loss_fn=torch.nn.CrossEntropyLoss(),
        data_loaders=loaders,
        clip_model=clip_model,
    )
    trainer.train()
    model.save_pretrained(args.output)


if __name__ == "__main__":
    main()
