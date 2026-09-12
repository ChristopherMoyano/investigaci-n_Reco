"""
KNN_paralelismo_optimizado.py

Cambios respecto a la version original:
  1. RandomUnderSampler ya no copia la matriz completa (676 cols) -> se le pasan
     solo indices (1 col). Evita el MemoryError de "check_array".
  2. X_otro ya NO se re-evalua al 100% en cada una de las 900 iteraciones.
     Se muestrea una fraccion configurable (OTRO_SAMPLE_FRAC) distinta cada vez.
     Esto es el cambio que mas tiempo ahorra: el costo de predict en KNN por
     fuerza bruta escala linealmente con el tamano de X_test.
  3. Se fuerza n_jobs=1 y se limitan los hilos de BLAS/OpenMP por engine, para
     evitar que 12 engines compitan entre si por los mismos nucleos de CPU
     (sobre-suscripcion silenciosa que ralentiza todo sin que se note en logs).
  4. Checkpoints: cada engine guarda resultados parciales a disco despues de
     cada iteracion de "num_under", con nombre unico por combinacion de
     parametros. Si el proceso se cae, no se pierde el trabajo ya hecho.
  5. Logging con timestamps (flush=True) para saber si avanza o esta trabado.
  6. Barra de progreso en el proceso principal (a nivel de combinaciones)
     usando tqdm + ar.progress de ipyparallel.
  7. try/except alrededor del loop pesado: si una iteracion falla, se registra
     el error y se continua, en vez de perder toda la combinacion.
  8. Modo DRY_RUN para probar con parametros chicos antes de lanzar la corrida
     completa (recomendado: correr esto primero, siempre).

Antes de lanzar la corrida completa en serio:
  - Deja DRY_RUN = True y ejecuta. Debería terminar en 1-3 minutos.
  - Revisa los tiempos que imprime por iteracion para estimar el total real.
  - Recien ahi cambia a DRY_RUN = False.
"""

import os
import time
import pickle
import glob
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import ipyparallel as ipp
from tqdm import tqdm

# ============================================================
# CONFIGURACION
# ============================================================

DRY_RUN = False          # <-- IMPORTANTE: dejar True para la primera prueba
CHECKPOINT_EVERY = 1    # guardar checkpoint cada N iteraciones de num_under
OTRO_SAMPLE_FRAC = 1 # fraccion de X_otro a evaluar en cada iteracion (antes: 1.0 = 100%)

num_cluster = 12
n_neighbors_arr = [1, 3, 11, 15]
weights_arr = ["uniform", "distance"]
p_arr = [1, 2]
param_list = list(product(n_neighbors_arr, weights_arr, p_arr))

if DRY_RUN:
    print("=== DRY RUN ACTIVADO: usando parametros reducidos para probar ===")
    param_list = param_list[:2]     # solo 2 combinaciones
    NUM_UNDER = 2
    NUM_EXP = 2
else:
    NUM_UNDER = 30
    NUM_EXP = 30

tag = "vuln_embeddings"
tag2 = "texto+num"

os.makedirs("pickle", exist_ok=True)
os.makedirs("pickle/checkpoints", exist_ok=True)

# ============================================================
# CLUSTER
# ============================================================

start = time.time()

cluster = ipp.Cluster(n=num_cluster)
cluster.start_cluster_sync()
rc = cluster.connect_client_sync()
print("Engines conectados:", rc.ids)
rc.wait_for_engines(num_cluster)
dview = rc[:]

# ============================================================
# FUNCION QUE CORRE EN CADA ENGINE
# ============================================================

def asignaciones(params, num_under, num_exp, otro_sample_frac, checkpoint_every, tag, tag2):
    import os
    # Limitar hilos de BLAS/OpenMP ANTES de importar numpy/sklearn en el engine,
    # para que los 12 engines no se peleen por los mismos nucleos de CPU.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    import numpy as np
    import pandas as pd
    import time
    import pickle
    import gc
    from datetime import datetime
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn import neighbors
    from imblearn.under_sampling import RandomUnderSampler

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}][k={k_v},w={w_v},p={p_v}] {msg}", flush=True)

    k_v, w_v, p_v = params
    combo_id = f"k{k_v}_w{w_v}_p{p_v}"

    path1 = 'dataset_' + tag + '.csv'
    Data_df = pd.read_csv(path1)
    Data_df = Data_df.drop(columns=[
        "version", "published", "date_reserved", "cve_id", "cwe_name", "published_year",
        "cwe_main", "ap_classOrName", "mo_phases",
        'comm_router', 'comm_switch', 'comm_access_point', 'comm_firewall', 'comm_vpn',
        'comm_openvpn', 'comm_modem', 'comm_cisco_ios', 'comm_cisco_ios_xe', 'comm_cisco_nx_os',
        'comm_junos', 'comm_routeros', 'comm_pan_os', 'comm_fortios', 'comm_sonicos',
        'comm_checkpoint_gaia', 'comm_6wind', 'comm_barracuda', 'comm_f5', 'comm_kemp_technologies',
        'comm_netapp', 'comm_netcracker', 'comm_pica8', 'comm_extremenetworks', 'comm_alcatel',
        'comm_hpe', 'comm_aruba', 'comm_avm', 'comm_calix', 'comm_ericsson', 'comm_cisco',
        'comm_d_link', 'comm_dlink', 'comm_comcast', 'comm_fiberhome', 'comm_fortinet',
        'comm_huawei', 'comm_juniper', 'comm_mikrotik', 'comm_nec', 'comm_silver_peak',
        'comm_tp_link', 'comm_zte', 'comm_cambium', 'comm_broadcom', 'comm_realtek', 'comm_arista',
        'comm_linksys', 'comm_exinda', 'comm_infoblox', 'comm_lte', 'comm_5g', 'comm_gsm',
        'comm_umts', 'comm_wifi', 'comm_bluetooth', 'comm_sip', 'comm_voip', 'comm_pbx',
        'comm_sdn', 'comm_nfv', 'comm_iot', 'comm_ethernet', 'comm_dsl', 'comm_arp', 'comm_hdlc',
        'comm_lldp', 'comm_mac', 'comm_ppp', 'comm_stp', 'comm_vlan', 'comm_isis', 'comm_mpls',
        'comm_nat', 'comm_vrrp', 'comm_ip', 'comm_icmp', 'comm_rip', 'comm_ospf', 'comm_tcp',
        'comm_udp', 'comm_rpc', 'comm_ssl', 'comm_tls', 'comm_dhcp', 'comm_dns', 'comm_http',
        'comm_snmp', 'comm_ftp', 'comm_ssh', 'percentile', 'epss', 'kev_dateAdded', 'delta_fecha',
    ], errors='ignore')
    Data_df = Data_df.dropna()

    y_full = Data_df["has_kev"].to_numpy(dtype=np.int8)
    X_full = Data_df.drop(columns=["has_kev"]).to_numpy(dtype=np.float32)
    n_total = len(y_full)

    log(f"Dataset cargado: {X_full.shape}, positivos={y_full.sum()}, negativos={n_total - y_full.sum()}")

    models = []
    confusion_matrices_test = []
    confusion_matrices_train = []

    t0 = time.perf_counter()
    errores = []

    for j in range(num_under):
        try:
            # --- Undersampling pasando solo INDICES, no la matriz completa ---
            idx_all = np.arange(n_total).reshape(-1, 1)
            rus = RandomUnderSampler(sampling_strategy=1)
            idx_res, y_res = rus.fit_resample(idx_all, y_full)
            idx_res = idx_res.ravel()

            mask = np.ones(n_total, dtype=bool)
            mask[idx_res] = False
            idx_otro_full = np.where(mask)[0]

            X_res = X_full[idx_res]
            # y_res ya viene de fit_resample

            for i in range(num_exp):
                # --- Muestreo de X_otro (en vez de usar el 100% siempre) ---
                # Reduce drasticamente el costo de predict en KNN por fuerza bruta.
                n_sample = max(1, int(len(idx_otro_full) * otro_sample_frac))
                idx_otro_sample = np.random.choice(idx_otro_full, size=n_sample, replace=False)
                X_otro = X_full[idx_otro_sample]
                y_otro = y_full[idx_otro_sample]

                X_train, X_test, y_train, y_test = train_test_split(
                    X_res, y_res, test_size=0.3
                )
                X_test = np.vstack([X_test, X_otro])
                y_test = np.concatenate([y_test, y_otro])

                clf = neighbors.KNeighborsClassifier(
                    n_neighbors=k_v,
                    weights=w_v,
                    p=p_v,
                    metric="minkowski",
                    n_jobs=1,  # evita que cada engine sub-paralelice y sature CPU
                )

                iter_t0 = time.perf_counter()
                y_pred = clf.fit(X_train, y_train).predict(X_test)
                y_pred_entrena = clf.predict(X_train)
                iter_t1 = time.perf_counter()

                conf_mat_train = confusion_matrix(y_train, y_pred_entrena)
                conf_mat_test = confusion_matrix(y_test, y_pred)

                models.append(clf)
                confusion_matrices_test.append(conf_mat_test)
                confusion_matrices_train.append(conf_mat_train)

                if i == 0:
                    log(f"under={j+1}/{num_under} exp={i+1}/{num_exp} "
                        f"X_train={X_train.shape} X_test={X_test.shape} "
                        f"tiempo_iter={iter_t1-iter_t0:.2f}s")

            del X_res, y_res, X_otro, y_otro, X_train, X_test, y_train, y_test
            gc.collect()

        except Exception as e:
            msg = f"ERROR en under={j}: {type(e).__name__}: {e}"
            log(msg)
            errores.append(msg)
            continue

        # --- checkpoint parcial a disco ---
        if (j + 1) % checkpoint_every == 0:
            ckpt_path = f"pickle/checkpoints/ckpt_{combo_id}_{tag2}_under{j+1}.pckl"
            with open(ckpt_path, "wb") as f:
                pickle.dump({
                    "vecinos": k_v, "weights": w_v, "p": p_v,
                    "under_completados": j + 1,
                    "modelos": models,
                    "conf_matrix_test": confusion_matrices_test,
                    "conf_matrix_train": confusion_matrices_train,
                    "errores": errores,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            log(f"checkpoint guardado ({j+1}/{num_under} completados)")

    t1 = time.perf_counter()
    log(f"TERMINADO combo. tiempo_total={t1-t0:.1f}s errores={len(errores)}")

    return {
        "vecinos": k_v,
        "weights": w_v,
        "p": p_v,
        "time_train": t1 - t0,
        "modelos": models,
        "conf_matrix_test": confusion_matrices_test,
        "conf_matrix_train": confusion_matrices_train,
        "errores": errores,
    }


# ============================================================
# LANZAR EN PARALELO CON BARRA DE PROGRESO
# ============================================================

print(f"\nLanzando {len(param_list)} combinaciones en {num_cluster} engines...")
print(f"NUM_UNDER={NUM_UNDER}  NUM_EXP={NUM_EXP}  OTRO_SAMPLE_FRAC={OTRO_SAMPLE_FRAC}")
print("Revisa la carpeta pickle/checkpoints/ para ver avance real de cada engine.\n")

ar = dview.map_async(
    asignaciones,
    param_list,
    [NUM_UNDER] * len(param_list),
    [NUM_EXP] * len(param_list),
    [OTRO_SAMPLE_FRAC] * len(param_list),
    [CHECKPOINT_EVERY] * len(param_list),
    [tag] * len(param_list),
    [tag2] * len(param_list),
)

# Barra de progreso a nivel de "combinaciones completas" (16 en total).
# Para ver avance DENTRO de cada combinacion, revisa los logs impresos
# por cada engine (log(...) con flush=True) y los checkpoints en disco.
with tqdm(total=len(param_list), desc="Combinaciones completadas") as pbar:
    last = 0
    while not ar.ready():
        done = ar.progress
        if done > last:
            pbar.update(done - last)
            last = done
        time.sleep(5)
    pbar.update(len(param_list) - last)

print(f"\nTiempo total de ejecucion: {(time.time()-start)/3600:.2f} horas")

# ============================================================
# RECOLECTAR RESULTADOS (con manejo de fallas por combo)
# ============================================================

results = []
try:
    raw_results = ar.get()
except Exception as e:
    print("ADVERTENCIA: al menos una combinacion fallo por completo.")
    print(e)
    raw_results = ar.get(return_exceptions=True) if hasattr(ar, "get") else []

for i, r in enumerate(raw_results):
    if isinstance(r, Exception):
        print(f"\n--- ERROR TOTAL en combo {param_list[i]} ---")
        print(r)
    else:
        results.append(r)

print(f"\nCombinaciones exitosas: {len(results)} / {len(param_list)}")

# ============ almacenar resultados ===============
with open(f"pickle/KNN_par_results_{tag}_{tag2}.pckl", "wb") as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

all_models = []
all_conf_train = []
all_conf_test = []
for res in results:
    all_models.extend(res["modelos"])
    all_conf_train.extend(res["conf_matrix_train"])
    all_conf_test.extend(res["conf_matrix_test"])

with open(f"pickle/models_KNN_{tag}_{tag2}.pckl", "wb") as f:
    pickle.dump(all_models, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"pickle/matrices_conf_train_KNN_{tag}_{tag2}.pckl", "wb") as f:
    pickle.dump(all_conf_train, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"pickle/matrices_conf_test_KNN_{tag}_{tag2}.pckl", "wb") as f:
    pickle.dump(all_conf_test, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\nTotal modelos guardados: {len(all_models)}")
print("Listo.")