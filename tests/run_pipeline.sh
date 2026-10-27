#!/bin/bash
# Prueba de humo de Thought2Text: ejecuta el pipeline COMPLETO
# (etapa 1 -> etapas 2 y 3 -> inferencia) con los scripts del repo,
# sobre datos y modelos en miniatura generados al momento. Corre en CPU, sin GPU
# y sin descargar modelos, en unos minutos.
#
# No mide la calidad del modelo (los modelos son de juguete): solo comprueba
# que el código corre de punta a punta. Si termina con "PIPELINE COMPLETO OK", pasó.
#
# Uso, desde la raíz del repo:  bash tests/run_pipeline.sh [python] [carpeta_temporal]
set -u
PY=${1:-python}
FX=$(realpath -m "${2:-/tmp/t2t_prueba}")
cd "$(dirname "$0")/.." || exit 1
rm -rf "$FX"

export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1  # todo es local

echo "##### Preparando datos y modelos de prueba en $FX"
$PY tests/make_fixtures.py "$FX" || { echo "FALLO al crear los datos de prueba"; exit 1; }

COMMON="--eeg_dataset $FX/data/block/eeg.pth --splits_path $FX/data/block/splits.pth --image_dir $FX/data/images/ --clip_model $FX/models/clip"

echo "##### ETAPA 1: train_eeg_classifier.py"
$PY train_eeg_classifier.py $COMMON --output "$FX/out/encoder" --device cpu \
    --batch_size 4 --num_epochs 3 --learning_rate 1e-3 --logging_steps 5 \
    || { echo "FALLO ETAPA 1"; exit 1; }

echo "##### ETAPAS 2 y 3: finetune_llm.py"
$PY finetune_llm.py $COMMON --eeg_encoder_path "$FX/out/encoder" --output "$FX/out/eegllm" \
    --llm_backbone_name_or_path "$FX/models/llm" --saved_pretrained_model_path "$FX/out/stage2" \
    --device cpu --batch_size 2 --gradient_accumulation_steps 1 --num_epochs_image 1 --num_epochs_eeg 1 \
    --logging_steps 5 || { echo "FALLO ETAPAS 2 y 3"; exit 1; }

echo "##### INFERENCIA: inference.py"
$PY inference.py --eeg_dataset "$FX/data/block/eeg.pth" --splits_path "$FX/data/block/splits.pth" \
    --image_dir "$FX/data/images/" --model_path "$FX/out/eegllm" --dest "$FX/out/results.csv" --device cpu \
    || { echo "FALLO INFERENCIA"; exit 1; }

[ -s "$FX/out/results.csv" ] || { echo "FALLO: inference.py no generó results.csv"; exit 1; }
echo "##### PIPELINE COMPLETO OK"
