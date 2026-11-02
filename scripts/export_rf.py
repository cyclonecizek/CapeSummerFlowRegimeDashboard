#!/usr/bin/env python3
"""
Export a scikit-learn RandomForestClassifier (.sav / joblib) to compact JSON
that the Cape page evaluates in the browser.

  python scripts/export_rf.py model.sav data/rf_xmr10z.json

Run it with the scikit-learn version the model was trained with (1.3.2 here).
Per tree: f = feature index (-1 at leaves), t = threshold, r = right-child index
(the left child is always the next node, since scikit-learn stores trees depth-first),
v = class-1 probability at leaves. predict_proba = mean of leaf probabilities.
"""
import json, sys
import joblib, numpy as np

def main(src, dst):
    m = joblib.load(src)
    out = {"type": "RandomForestClassifier", "features": [str(x) for x in m.feature_names_in_],
           "classes": [float(c) for c in m.classes_], "n_estimators": len(m.estimators_),
           "max_depth": int(max(e.tree_.max_depth for e in m.estimators_)),
           "oob_score": round(float(getattr(m, "oob_score_", float("nan"))), 4),
           "importances": [round(float(x), 4) for x in m.feature_importances_], "trees": []}
    pos = list(m.classes_).index(1.0) if 1.0 in list(m.classes_) else 1
    for e in m.estimators_:
        t = e.tree_
        f, th, r, v = [], [], [], []
        for i in range(t.node_count):
            if t.children_left[i] == -1:
                cnt = t.value[i][0]; f.append(-1); th.append(0); r.append(-1); v.append(round(float(cnt[pos] / cnt.sum()), 5))
            else:
                assert t.children_left[i] == i + 1, "expected depth-first layout"
                f.append(int(t.feature[i])); th.append(round(float(t.threshold[i]), 5)); r.append(int(t.children_right[i])); v.append(0)
        out["trees"].append({"f": f, "t": th, "r": r, "v": v})
    with open(dst, "w") as fh: json.dump(out, fh, separators=(",", ":"))
    print(f"wrote {dst}: {len(out['trees'])} trees, features {out['features']}")

if __name__ == "__main__":
    if len(sys.argv) != 3: sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
