# src/train_tf.py
import os, argparse, json, pandas as pd, numpy as np, tensorflow as tf, keras
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ====== Args ======
p = argparse.ArgumentParser()
p.add_argument("--train_path", type=str, required=True)  # data asset (9k xlsx o csv)
p.add_argument("--output_dir", type=str, default="outputs")  # AML sube 'outputs' al run
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--batch_size", type=int, default=24)
p.add_argument("--max_len", type=int, default=128)
args = p.parse_args()

os.environ["USE_TF"] = "1"; os.environ["TRANSFORMERS_NO_TORCH"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ====== Config/labels ======
MODEL_NAME = "distilbert-base-multilingual-cased"
LABELS = ["ruso", "ucraniano", "neutro"]
LABEL2ID = {l:i for i,l in enumerate(LABELS)}
ID2LABEL = {i:l for l,i in LABEL2ID.items()}

# ====== Carga 9k ======
if args.train_path.endswith(".xlsx"):
    df = pd.read_excel(args.train_path, engine="openpyxl")
else:
    df = pd.read_csv(args.train_path)

df["comment"] = df["comment"].astype(str)
df["condiciones_cuenta"] = df["condiciones_cuenta"].astype(str).str.strip().str.lower()
def clean_text(s):
    import re
    s = re.sub(r"http\S+"," ",s); s = re.sub(r"@[A-Za-z0-9_]+"," ",s)
    s = re.sub(r"[^A-Za-zÀ-ÿ0-9\s]"," ",s); s = re.sub(r"\s+"," ",s).strip().lower()
    return s
df["comment_clean"] = df["comment"].apply(clean_text)
df["text_with_ctx"] = ("[canal:"+df["condiciones_cuenta"]+"] "+df["comment_clean"]).str.strip()
y = df["label_final"].astype(str).str.strip().str.lower().map(LABEL2ID)
mask = y.notna()
X = df.loc[mask, "text_with_ctx"].tolist()
y = y.loc[mask].astype(int).values

Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ====== Tokenizer/Datasets ======
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
def encode(texts):
    enc = tok(texts, truncation=True, padding="max_length", max_length=args.max_len)
    return {k: np.array(v) for k,v in enc.items()}

BATCH_TRAIN = args.batch_size; BATCH_VAL = max(2*args.batch_size, 64)
enc_tr = encode(Xtr); enc_va = encode(Xva)
train_ds = tf.data.Dataset.from_tensor_slices((enc_tr, ytr, np.ones_like(ytr))).shuffle(2048).batch(BATCH_TRAIN).prefetch(tf.data.AUTOTUNE)
val_ds   = tf.data.Dataset.from_tensor_slices((enc_va, yva, np.ones_like(yva))).batch(BATCH_VAL).prefetch(tf.data.AUTOTUNE)
train_ds = train_ds.map(lambda enc,y,sw: (enc, tf.cast(y, tf.int32), tf.cast(sw, tf.float32)))
val_ds   = val_ds.map(lambda enc,y,sw: (enc, tf.cast(y, tf.int32), tf.cast(sw, tf.float32)))

# ====== Modelo ======
model = TFAutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
)

# Warm-up 1 epoch congelado (acelera), luego fine-tune
try:
    model.distilbert.trainable = False
except:
    pass

optimizer = keras.optimizers.Adam(learning_rate=3e-5)
loss      = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics   = [keras.metrics.SparseCategoricalAccuracy(name="acc")]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)

try:
    model.distilbert.trainable = True
except:
    pass
model.compile(optimizer=keras.optimizers.Adam(3e-5), loss=loss, metrics=metrics)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
    keras.callbacks.CSVLogger(os.path.join(args.output_dir, "train_log.csv"))
]
history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs-1, verbose=1, callbacks=callbacks)

# ====== Eval y guardado ======
logits = model.predict(val_ds.map(lambda enc,y,sw: (enc,y)), verbose=0).logits
yhat = logits.argmax(axis=1)
rep = classification_report(yva, yhat, target_names=LABELS, digits=3)
print(rep)
with open(os.path.join(args.output_dir,"val_report.txt"),"w",encoding="utf-8") as f: f.write(rep)

save_dir = os.path.join(args.output_dir, "tf_distilmbert_stance_export")
model.save_pretrained(save_dir)
tok.save_pretrained(save_dir)
print("✅ saved to", save_dir)
