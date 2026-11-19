import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/raw/kredit_sumatera.csv")
df["rasio_angsuran"] = df.angsuran_bulanan / df.penghasilan_bulanan
num = ["usia", "tanggungan", "lama_usaha_tahun", "penghasilan_bulanan",
       "ragam_penghasilan", "plafon_diajukan", "tenor_bulan", "bunga_tahunan",
       "rasio_angsuran", "rata_saldo_90h", "jumlah_tunggakan_12b"]
X = df[num].join(pd.get_dummies(df[["segmen", "wilayah", "kanal"]]))
y = df.gagal_bayar
latih = df.periode <= "2023-12"
model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.06,
        max_iter=250, random_state=42).fit(X[latih], y[latih])

print("== Langkah 1: AUC per bulan 2024")
baris = []
for periode in sorted(df.loc[df.periode > "2023-12", "periode"].unique()):
    m = df.periode == periode
    try:
        auc = roc_auc_score(y[m], model.predict_proba(X[m])[:, 1])
    except ValueError:
        auc = float("nan")
    baris.append({"periode": periode, "n": int(m.sum()), "auc": auc, "laju_gagal": float(y[m].mean())})
hasil = pd.DataFrame(baris)
print(hasil.round(4).to_string(index=False))
hasil.to_csv("laporan_auc_bulanan.csv", index=False)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(hasil.periode, hasil.auc, marker="o")
    plt.xticks(rotation=45); plt.ylabel("AUC"); plt.title("AUC per bulan, model dilatih sampai 2023-12")
    plt.tight_layout(); plt.savefig("laporan_auc_bulanan.png", dpi=120)
    print("grafik tersimpan: laporan_auc_bulanan.png")
except ImportError:
    print("matplotlib belum terpasang: pip install matplotlib, lalu jalankan ulang")

print("\n== Langkah 2a: komposisi segmen awal vs akhir")
awal = df[df.periode <= "2022-06"]
akhir = df[df.periode >= "2024-07"]
print(pd.DataFrame({"awal": awal.segmen.value_counts(normalize=True),
                    "akhir": akhir.segmen.value_counts(normalize=True)}).round(3))

print("\n== Langkah 2b: AUC per segmen (uji > 2024-06)")
uji = df.periode > "2024-06"
skor = pd.Series(model.predict_proba(X[uji])[:, 1], index=df[uji].index)
for seg, g in df[uji].groupby("segmen"):
    print(f"{seg:<18} n={len(g):5d} auc={roc_auc_score(y.loc[g.index], skor.loc[g.index]):.3f}")
