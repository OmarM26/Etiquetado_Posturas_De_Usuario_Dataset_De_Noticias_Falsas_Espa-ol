import os
import json
import pandas as pd

def load_labels(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Subtask A: id_tweet -> label
    return {str(k): (v or "").lower() for k, v in data.items()}

def load_tweet(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_structure(struct_path):
    with open(struct_path, "r", encoding="utf-8") as f:
        struct = json.load(f)

    
    root_id = str(next(iter(struct.keys())))

    parent_of = {}  # child_id -> parent_id

    def walk(node, parent):
        # node puede ser dict, list, str/int o vacío
        if isinstance(node, dict):
            for k, v in node.items():
                kid = str(k)
                if parent is not None:
                    parent_of[kid] = parent
                walk(v, kid)
        elif isinstance(node, list):
            for item in node:
                walk(item, parent)
        elif isinstance(node, (str, int)):
            kid = str(node)
            if parent is not None:
                parent_of[kid] = parent
        else:
            # None, bool, etc. -> ignorar
            pass

    walk(struct[root_id], root_id)
    return parent_of, root_id

def build_paths(parent_of, rid, root_id):
    # profundidad = nº de aristas desde root hasta rid
    depth = 0
    path = [rid]
    cur = rid
    while cur != root_id and cur in parent_of:
        cur = parent_of[cur]
        path.append(cur)
        depth += 1
    path_ids = ">".join(reversed(path))  # root>...>rid
    parent_id = parent_of.get(rid, root_id)
    return parent_id, depth, path_ids

def parse_event(event_dir, labels, event_name):
    rows = []
    threads = [d for d in os.listdir(event_dir) if os.path.isdir(os.path.join(event_dir, d))]
    for th in threads:
        th_path = os.path.join(event_dir, th)
        src_dir = os.path.join(th_path, "source-tweet")
        rep_dir = os.path.join(th_path, "replies")
        struct_path = os.path.join(th_path, "structure.json")

        if not (os.path.exists(src_dir) and os.path.exists(rep_dir) and os.path.exists(struct_path)):
            continue

        # raíz
        try:
            src_file = os.listdir(src_dir)[0]
        except IndexError:
            continue
        root = load_tweet(os.path.join(src_dir, src_file))
        root_id = str(root.get("id_str") or root.get("id") or "")
        root_text = root.get("text") or root.get("full_text") or ""

        # estructura: padres, profundidad y rutas
        parent_of, _root_check = parse_structure(struct_path)
        # algunos dumps tienen id como int/str; forzamos a str
        parent_of = {str(k): str(v) for k, v in parent_of.items()}

        # indexamos todos los json de respuestas para poder leer parent_text
        id2text = {root_id: root_text}
        for rf in os.listdir(rep_dir):
            rj = load_tweet(os.path.join(rep_dir, rf))
            tid = str(rj.get("id_str") or rj.get("id") or "")
            ttext = rj.get("text") or rj.get("full_text") or ""
            id2text[tid] = ttext

        # recorre todas las replies del directorio (todo el árbol)
        for rf in os.listdir(rep_dir):
            rj = load_tweet(os.path.join(rep_dir, rf))
            rid = str(rj.get("id_str") or rj.get("id") or "")
            if rid not in labels:
                continue  # solo tweets con etiqueta SDQC en Subtask A

            rtext = rj.get("text") or rj.get("full_text") or ""
            parent_id, depth, path_ids = build_paths(parent_of, rid, root_id)
            parent_text = id2text.get(parent_id, "")

            # arma contexto padre+raíz, y path_text opcional (camino sin el actual)
            context_text = (parent_text + " [SEP] " + root_text).strip() if parent_text else root_text

            # construir path_text con los ancestros (root..parent) usando path_ids:
            path_text = ""
            if path_ids:
                anc = path_ids.split(">")[:-1]  # todas menos el actual
                if anc:
                    path_text = " [EOU] ".join([id2text.get(tid, "") for tid in anc])

            
            rows.append({
                "event": event_name,
                "tweet_id": rid,
                "text": rtext,
                "label_sdqc": labels[rid],
                "root_id": root_id,
                "root_text": root_text,
                "parent_id": parent_id,
                "parent_text": parent_text,
                "depth": depth,
                "path_ids": path_ids,
                "context_text": context_text,
                "path_text": path_text

            })
    return rows

def parse_all(base_dir, labels_path):
    labels = load_labels(labels_path)
    all_rows = []
    for ev in os.listdir(base_dir):
        ev_dir = os.path.join(base_dir, ev)
        if not os.path.isdir(ev_dir):
            continue
        print(f"Evento: {ev}")
        rows = parse_event(ev_dir, labels, ev)
        print(f"  -> {len(rows)} ejemplos")
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)

if __name__ == "__main__":
    # Ajusta estas rutas a tu máquina
    base = "C:/Users/omarm/Downloads/estado del arte papers/datasets/semeval2017-task8-dataset/rumoureval-data"
    labels_train = "../semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json"
    labels_dev   = "../datasets/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-dev.json"

    df_train = parse_all(base, labels_train)
    df_dev   = parse_all(base, labels_dev)

    print("Train ejemplos:", len(df_train))
    print("Dev ejemplos:", len(df_dev))

    # Guarda en el directorio actual; si quieres
    df_train.to_csv("rumoureval_train.csv", index=False, encoding="utf-8-sig")
    df_dev.to_csv("rumoureval_dev.csv", index=False, encoding="utf-8-sig")
    print("CSV guardados")
