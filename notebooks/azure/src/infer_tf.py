# src/infer_tf.py
import os, argparse, pandas as pd, numpy as np, tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

p = argparse.ArgumentParser()
p.add_argument("--input_path", type=str, required=True)   # data asset 300k
p.add_argument("--model_dir", type=str, required=True)    # carpeta export (save_pretrained)
p.add_argument("--out_csv", type=str, default="outputs/predicciones_full_new.csv")
p.add_argument("--chunk_rows", type=int, default=50000)
p.add_argument("--batch_infer", type=int, default=1024)
p.add_argument("--max_len", type=int, default=128)
p.add_argument("--umbral_proba", type=float, default=0.55)
p.add_argument("--umbral_margen", type=float, default=0.15)
args = p.parse_args()

LABELS = ["ruso","ucraniano","neutro"]

def clean_text(s):
    import re
    s = re.sub(r"http\S+"," ",s); s = re.sub(r"@[A-Za-z0-9_]+"," ",s)
    s = re.sub(r"[^A-Za-zÀ-ÿ0-9\s]"," ",s); s = re.sub(r"\s+"," ",s).strip().lower()
    return s

# ——— tus reglas rápidas (puedes pegar aquí tu rule_classifier exacta) ———
PRAISE = ["buen canal","buen análisis","buen analisis","me gusta el canal","excelente video","gran resumen","muy claro","te felicito","buen contenido","gran trabajo"]
ANTI_OCC = ["otan","cnn","occidente","eeuu","estados unidos","propaganda occidental","ue propaganda","nato"]
ANTI_RUS = ["putin asesino","dictador","rusia invasora","invasion rusa","propaganda rusa","kremlin miente","criminal de guerra"]
def contains_any(s, bag): s=s.lower(); return any(k in s for k in bag)
def rule_classifier(row):
    text = str(row["comment"]).lower()
    canal = str(row["condiciones_cuenta"]).strip().lower()
    if contains_any(text, PRAISE) and canal in {"pro-ucraniano","pro-ruso"}:
        return ("ucraniano" if canal=="pro-ucraniano" else "ruso"), "regla-elogio-canal"
    if canal == "pro-ucraniano" and contains_any(text, ANTI_OCC):
        return "ruso", "regla-anti-occidente-en-canal-pro-ucr"
    if canal == "pro-ruso" and contains_any(text, ANTI_RUS):
        return "ucraniano", "regla-anti-rusia-en-canal-pro-ruso"
    return None, None
# ————————————————————————————————————————————————————————————————

tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
model = TFAutoModelForSequenceClassification.from_pretrained(args.model_dir)

def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# reset CSV
os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
if os.path.exists(args.out_csv): os.remove(args.out_csv)

processed = 0; milestone = 50_000
for df_part in pd.read_csv(args.input_path, chunksize=args.chunk_rows):
    # normalización
    df = df_part.copy()
    df["comment"] = df["comment"].astype(str)
    df["condiciones_cuenta"] = df["condiciones_cuenta"].astype(str).str.strip().str.lower()
    df["comment_clean"] = df["comment"].apply(clean_text)
    df["text_with_ctx"] = ("[canal:"+df["condiciones_cuenta"]+"] "+df["comment_clean"]).str.strip()

    # reglas
    df["label_rule"] = None; df["regla_aplicada"] = None
    for i in range(len(df)):
        lab, rname = rule_classifier(df.iloc[i])
        df.iat[i, df.columns.get_loc("label_rule")] = lab
        df.iat[i, df.columns.get_loc("regla_aplicada")] = rname

    is_rule = df["label_rule"].isin(LABELS)
    to_pred = ~is_rule

    idxs  = df.loc[to_pred].index.tolist()
    texts = df.loc[to_pred, "text_with_ctx"].tolist()

    # NN por lotes (GPU-ready)
    top1_lab_all, pmax_all, marg_all, ent_all = [], [], [], []
    for chunk_txt in batched(texts, args.batch_infer):
        enc = tok(chunk_txt, truncation=True, padding=True, max_length=args.max_len, return_tensors="tf")
        out = model(enc, training=False)
        logits = out.logits.numpy()
        probas = tf.nn.softmax(logits, axis=1).numpy()
        top1   = probas.argmax(axis=1)
        pmax   = probas[np.arange(probas.shape[0]), top1]
        sortp  = -np.sort(-probas, axis=1)
        marg   = sortp[:,0] - sortp[:,1]
        entr   = -(probas * (np.log(probas + 1e-12))).sum(axis=1)

        top1_lab_all.append(np.array(LABELS)[top1])
        pmax_all.append(pmax); marg_all.append(marg); ent_all.append(entr)

    if len(top1_lab_all):
        df.loc[idxs, "label_ml"]     = np.concatenate(top1_lab_all)
        df.loc[idxs, "ml_proba_max"] = np.concatenate(pmax_all)
        df.loc[idxs, "ml_margen"]    = np.concatenate(marg_all)
        df.loc[idxs, "ml_entropia"]  = np.concatenate(ent_all)
    else:
        for c in ["label_ml","ml_proba_max","ml_margen","ml_entropia"]: df[c] = np.nan

    # abstención
    def decide_row(row):
        lab = row.get("label_ml",""); p  = row.get("ml_proba_max",0.0); mg = row.get("ml_margen",0.0)
        if isinstance(lab,float): lab=""
        return (lab, "automatica-nn-tf") if (p >= args.umbral_proba and mg >= args.umbral_margen) else ("","sin-clasificar")
    dec = df.apply(decide_row, axis=1, result_type="expand")
    df["label_final"] = dec[0]; df["clasificacion_origen"] = dec[1]
    df.loc[is_rule, "label_final"] = df.loc[is_rule, "label_rule"]
    df.loc[is_rule, "clasificacion_origen"] = "regla"

    # append CSV
    df.to_csv(args.out_csv, mode="a", header=not os.path.exists(args.out_csv), index=False, encoding="utf-8-sig")
    processed += len(df_part)
    if processed >= milestone:
        print(f"[LOG] Procesadas {processed:,} filas → {args.out_csv}")
        milestone += 50_000

print(f"✅ FIN. CSV en {args.out_csv}")
