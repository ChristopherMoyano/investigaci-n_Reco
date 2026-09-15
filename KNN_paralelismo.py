import numpy as np
import ipyparallel as ipp
from timeit import default_timer as timer
from itertools import product
import pandas as pd
import pickle
import os
from datetime import datetime
import glob
from tqdm import tqdm
import time

# ============================================
# TAG
# ============================================

#Profesor estos son los casos que faltan, solo tiene que descomentar el que quiera probar, y comentar los demas.


tag_df = "vuln_2020_2025_texto"
tag_num_tex = "texto"
tag_modelo = "KNN"


#tag_df = "vuln_2020_2025_num"
#tag_num_tex = "num"

#tag_df = "vuln_TF_IDF_2020_2025_num+texto"
#tag_num_tex = "num+texto"

#tag_df = "vuln_TF_IDF_2020_2025_texto"
#tag_num_tex = "texto"


# ============================================
# CONFIGURACION
# ============================================

DRY_RUN = False   

checkpoint_every = 5
n_neighbors_arr = [1, 3, 11, 15]
weights_arr = ["uniform", "distance"]
p_arr = [1, 2]
param_list = list(product(n_neighbors_arr, weights_arr, p_arr))

if DRY_RUN:
    print("=== DRY RUN ACTIVADO: probando pocas combinaciones antes de la corrida completa ===")
    param_list = param_list[:3]   
    num_exp = 3
else:
    num_exp = 50


num_cluster = len(param_list)

os.makedirs("pickle", exist_ok=True)
os.makedirs("pickle/checkpoints", exist_ok=True)
os.makedirs("pickle/best_models", exist_ok=True)   

# ===========================================
# CLUSTER
# ===========================================
start = time.time()

cluster = ipp.Cluster(n=num_cluster)
cluster.start_cluster_sync()
rc = cluster.connect_client_sync()
print("Engines conectados:", rc.ids)
rc.wait_for_engines(num_cluster)
dview = rc[:]


# ==========================================
# FUNCION QUE CORRE EN CADA ENGINE
# ==========================================

def asignaciones(params, num_exp, checkpoint_every, tag_df, tag_num_tex, tag_modelo):
    import os

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
    from sklearn import config_context

    n_neighbors_v, weights_v, p_v = params
    combo_id = f"n{n_neighbors_v}_w{weights_v}_p{p_v}"  # nombre compacto, sin espacios/comas raras

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}][{combo_id}] {msg}", flush=True)

    path = 'dataset_' + tag_df + '.csv'
    try:
        Data_df = pd.read_csv(path)
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
        y = Data_df["has_kev"].to_numpy(dtype=np.int8)
        X = Data_df.drop(columns=["has_kev"]).to_numpy(dtype=np.float32)
        n_total = len(y)
        log(f"Dataset cargado: {X.shape}, positivos={y.sum()}, negativos={n_total - y.sum()}")
    except Exception as e:
        log(f"ERROR CRITICO al cargar datos: {type(e).__name__}: {e}")
        return {
            "n_neighbors": n_neighbors_v, "weights": weights_v, "p": p_v,
            "conf_matrix_test": [], "conf_matrix_train": [],
            "errores": [f"fallo en carga de datos: {e}"],
            "best_model_path": None, "best_score": None,
        }

    confusion_matrices_test = []
    confusion_matrices_train = []
    errores = []

    # --- se inicializan ANTES del loop, este era el bug que impedia guardar el mejor modelo ---
    mejor_score = -1.0
    mejor_modelo = None

    ti = time.perf_counter()
    for i in range(num_exp):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

            clf = neighbors.KNeighborsClassifier(
                n_neighbors=n_neighbors_v,
                weights=weights_v,
                p=p_v,
                metric="minkowski",
                n_jobs=1
            )

            
            with config_context(working_memory=64):
                y_pred = clf.fit(X_train, y_train).predict(X_test)
                y_pred_entrena = clf.predict(X_train)

            conf_mat_train = confusion_matrix(y_train, y_pred_entrena)
            conf_mat_test = confusion_matrix(y_test, y_pred)

            confusion_matrices_test.append(conf_mat_test)
            confusion_matrices_train.append(conf_mat_train)

          
            score_actual = conf_mat_test[1, 1] / max(conf_mat_test[1, 1] + conf_mat_test[1, 0], 1)
            if score_actual > mejor_score:
                mejor_score = score_actual
                mejor_modelo = clf

            
            if (i + 1) % checkpoint_every == 0:
                ckpt_path = f"pickle/checkpoints/ckpt_{combo_id}_{tag_df}.pckl"
                with open(ckpt_path, "wb") as f:
                    pickle.dump({
                        "n_neighbors": n_neighbors_v, "weights": weights_v, "p": p_v,
                        "conf_matrix_test": confusion_matrices_test,
                        "conf_matrix_train": confusion_matrices_train,
                        "errores": errores,
                        "experimentos_completados": i + 1,
                        "mejor_score_hasta_ahora": mejor_score,
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
                log(f"checkpoint guardado ({i+1}/{num_exp}), mejor_score={mejor_score:.4f}")

        except Exception as e:
            msg = f"ERROR en exp={i}: {type(e).__name__}: {e}"
            log(msg)
            errores.append(msg)
            continue

    tf = time.perf_counter()


    best_model_path = None
    if mejor_modelo is not None:
        best_model_path = f"pickle/best_models/best_{tag_modelo}_{combo_id}_{tag_df}.pckl"
        with open(best_model_path, "wb") as f:
            pickle.dump({
                "n_neighbors": n_neighbors_v, "weights": weights_v, "p": p_v,
                "score": mejor_score,
                "modelo": mejor_modelo,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"mejor modelo guardado en disco: {best_model_path} (score={mejor_score:.4f})")

    X = None
    y = None
    mejor_modelo = None
    gc.collect()

    log(f"TERMINADO combo. tiempo total: {tf-ti:.1f}s errores={len(errores)}")

    
    return {
        "n_neighbors": n_neighbors_v,
        "weights": weights_v,
        "p": p_v,
        "time_train": tf - ti,
        "conf_matrix_test": confusion_matrices_test,
        "conf_matrix_train": confusion_matrices_train,
        "errores": errores,
        "best_model_path": best_model_path,
        "best_score": mejor_score if mejor_modelo is None else mejor_score,
    }


# ==================================================
# LANZAR EN PARALELO CON BARRA DE PROGRESO
# ==================================================

print(f"\nLanzando {len(param_list)} combinaciones en {num_cluster} engines...")
print(f"\nNUM_EXP ={num_exp}")
print("Revisa la carpeta pickle/checkpoints/ para ver avance real de cada engine.")
print("Los mejores modelos se guardan directo en pickle/best_models/ por cada engine.\n")

ar = dview.map_async(
    asignaciones,
    param_list,
    [num_exp] * len(param_list),
    [checkpoint_every] * len(param_list),
    [tag_df] * len(param_list),
    [tag_num_tex] * len(param_list),
    [tag_modelo] * len(param_list)
)

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


with open(f"pickle/{tag_modelo}_par_results_{tag_df}_{tag_num_tex}.pckl", "wb") as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

all_conf_train = []
all_conf_test = []
tabla_mejores = []   

for res in results:
    all_conf_train.extend(res["conf_matrix_train"])
    all_conf_test.extend(res["conf_matrix_test"])
    tabla_mejores.append({
        "n_neighbors": res["n_neighbors"],
        "weights": res["weights"],
        "p": res["p"],
        "best_score": res["best_score"],
        "best_model_path": res["best_model_path"],
    })

with open(f"pickle/matrices_conf_train_{tag_modelo}_{tag_df}_{tag_num_tex}.pckl", "wb") as f:
    pickle.dump(all_conf_train, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"pickle/matrices_conf_test_{tag_modelo}_{tag_df}_{tag_num_tex}.pckl", "wb") as f:
    pickle.dump(all_conf_test, f, protocol=pickle.HIGHEST_PROTOCOL)

# tabla resumen  facil de leer para saber cual modelo cargar despues
if tabla_mejores:
    tabla_df = pd.DataFrame(tabla_mejores).sort_values("best_score", ascending=False)
    tabla_df.to_csv(f"pickle/resumen_mejores_modelos_{tag_modelo}_{tag_df}.csv", index=False)
    print("\nResumen de mejores modelos por combinacion:")
    print(tabla_df.to_string(index=False))
else:
    print("\nADVERTENCIA: ninguna combinacion produjo resultados exitosos, "
          "no se genero tabla resumen. Revisa los errores impresos arriba.")

print("\nListo.")

