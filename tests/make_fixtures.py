"""Crea datos y modelos en miniatura con el MISMO formato que Thought2Text espera,
para probar los scripts del repo de punta a punta sin GPU ni descargas.

Uso (desde la raíz del repo): python tests/make_fixtures.py <carpeta_destino>
Lo usa tests/run_pipeline.sh; no hace falta llamarlo a mano.
Genera:
  <dest>/data/block/eeg.pth, <dest>/data/block/splits.pth
  <dest>/data/images/<wnid>/<img>{.JPEG,_sketch.JPEG,_caption.txt}
  <dest>/models/clip   (CLIP de visión diminuto, salida de 512 dim)
  <dest>/models/llm    (LLM tipo Llama diminuto + tokenizer con chat template)
"""
import os, sys, random
import torch
from PIL import Image

dest = os.path.abspath(sys.argv[1])
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # la carpeta padre de tests/
sys.path.insert(0, repo)
from constants import label_map, id2label  # noqa: E402

torch.manual_seed(0); random.seed(0)
name2wnid = {v: k for k, v in label_map.items()}
labels40 = [name2wnid[id2label[str(i)]] for i in range(40)]  # índice i <-> nombre de constants.id2label

CLASSES = ["dog", "airplane", "pizza", "piano"]
CAPTIONS = {"dog": "A brown dog sitting on the grass.", "airplane": "A white airplane parked on a runway.",
            "pizza": "A slice of pizza on a white plate.", "piano": "A black grand piano in a living room."}
COLORS = {"dog": (150, 90, 40), "airplane": (220, 220, 230), "pizza": (230, 150, 40), "piano": (20, 20, 20)}
IMGS_PER_CLASS, SUBJECTS = 6, 2

images, dataset = [], []
for c in CLASSES:
    wnid = name2wnid[c]; lab = labels40.index(wnid)
    os.makedirs(f"{dest}/data/images/{wnid}", exist_ok=True)
    # Firma por clase: oscilación de ~70 Hz (dentro de la banda 55–95 Hz del dataset real) con una
    # amplitud distinta por canal. Media cero, como los datos reales filtrados y estandarizados.
    amp = torch.randn(128, 1)
    freq = 60 + 8 * CLASSES.index(c)
    for k in range(IMGS_PER_CLASS):
        name = f"{wnid}_{k}"
        img = Image.new("RGB", (64, 64), COLORS[c])
        img.save(f"{dest}/data/images/{wnid}/{name}.JPEG"); img.save(f"{dest}/data/images/{wnid}/{name}_sketch.JPEG")
        with open(f"{dest}/data/images/{wnid}/{name}_caption.txt", "w") as f:
            f.write(f"<s> {CAPTIONS[c]} </s>\n")
        images.append(name)
        for s in range(1, SUBJECTS + 1):
            T = random.randint(470, 520)  # el Splitter exige 450 <= T <= 600
            t = torch.arange(T) / 1000.0  # 1000 Hz, como el dataset original
            eeg = torch.randn(128, T) + amp * torch.sin(2 * torch.pi * freq * t)
            dataset.append({"eeg": eeg, "label": lab, "image": len(images) - 1, "subject": s})

os.makedirs(f"{dest}/data/block", exist_ok=True)
torch.save({"dataset": dataset, "labels": labels40, "images": images}, f"{dest}/data/block/eeg.pth")
idx = list(range(len(dataset))); random.shuffle(idx)
n = len(idx)
torch.save({"splits": [{"train": idx[: int(.6 * n)], "val": idx[int(.6 * n): int(.8 * n)], "test": idx[int(.8 * n):]}]},
           f"{dest}/data/block/splits.pth")

# --- CLIP de visión diminuto (proyección a 512 como openai/clip-vit-base-patch32)
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection, CLIPImageProcessor
clip = CLIPVisionModelWithProjection(CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                                     num_attention_heads=2, image_size=224, patch_size=32,
                                                     projection_dim=512))
clip.save_pretrained(f"{dest}/models/clip"); CLIPImageProcessor().save_pretrained(f"{dest}/models/clip")

# --- LLM tipo Llama diminuto con tokenizer y chat template
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, LlamaConfig, LlamaForCausalLM
words = " ".join(CAPTIONS.values()) + " You are a helpful assistant. <image> Describe this image in one sentence: " \
        + " ".join(label_map.values())
t = Tokenizer(models.WordLevel(unk_token="<unk>")); t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
t.train_from_iterator([words] * 3, trainers.WordLevelTrainer(special_tokens=["<unk>", "<s>", "</s>", "system:", "user:", "assistant:"]))
tok = PreTrainedTokenizerFast(tokenizer_object=t, unk_token="<unk>", bos_token="<s>", eos_token="</s>")
tok.chat_template = ("{% for m in messages %}{{ m['role'] + ': ' + m['content'] + ' ' }}"
                     "{% if m['role']=='assistant' %}</s> {% endif %}{% endfor %}"
                     "{% if add_generation_prompt %}assistant: {% endif %}")
tok.save_pretrained(f"{dest}/models/llm")
llm = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=4,
                                   bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id))
llm.save_pretrained(f"{dest}/models/llm")
print(f"OK: {len(dataset)} registros EEG, {len(images)} imágenes, modelos en {dest}/models")
