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

tag_df = "vuln_TF_IDF_2020_2025_texto"
tag_num_tex ="texto"
tag_modelo = "LR"

# ============================================
# CONFIGURACION
# ============================================

num_exp = 50
num_cluster = 12
checkpoint_every = 5
penalty_arr = [None,"l2"]
c_var = [0.001, 1, 1000]
solver_arr = ["lbfgs","sag","saga"]
param_list = list(product(penalty_arr, c_var, solver_arr))


os.makedirs("pickle", exist_ok=True)
os.makedirs("pickle/checkpoints", exist_ok=True)

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

def asignaciones(params,num_exp,checkpoint_every,tag_df,tag_num_tex):
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
    from sklearn.linear_model import LogisticRegression

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}][solver ={solver_v},C={c_v},penalty = {penalty_v}] {msg}",flush=True)
    
    penalty_v, c_v, solver_v = params
    combo_id = f"solver ={solver_v},C={c_v},penalty = {penalty_v}"
    
    path = 'dataset_' + tag_df +'.csv'
    Data_df = pd.read_csv(path)
    Data_df = Data_df.drop(columns = ["version", "published", "date_reserved", "cve_id", "cwe_name", "published_year", "cwe_main", "ap_classOrName", "mo_phases",
                                      'comm_router',  'comm_switch',  'comm_access_point',  'comm_firewall', 'comm_vpn',
                                       'comm_openvpn',  'comm_modem',  'comm_cisco_ios',  'comm_cisco_ios_xe',  'comm_cisco_nx_os',  'comm_junos',  'comm_routeros',
                                       'comm_pan_os',  'comm_fortios',  'comm_sonicos',  'comm_checkpoint_gaia',  'comm_6wind',  'comm_barracuda',  'comm_f5',
                                       'comm_kemp_technologies',  'comm_netapp',  'comm_netcracker',  'comm_pica8',  'comm_extremenetworks',  'comm_alcatel',
                                       'comm_hpe', 'comm_aruba', 'comm_avm',  'comm_calix',  'comm_ericsson',  'comm_cisco',  'comm_d_link',  'comm_dlink',
                                       'comm_comcast',  'comm_fiberhome',  'comm_fortinet',  'comm_huawei',  'comm_juniper',  'comm_mikrotik',  'comm_nec',
                                       'comm_silver_peak',  'comm_tp_link',  'comm_zte',  'comm_cambium',  'comm_broadcom',  'comm_realtek',  'comm_arista',
                                       'comm_linksys', 'comm_exinda',  'comm_infoblox',  'comm_lte',  'comm_5g',  'comm_gsm',  'comm_umts',  'comm_wifi',
                                       'comm_bluetooth',  'comm_sip',  'comm_voip',  'comm_pbx',  'comm_sdn',  'comm_nfv',  'comm_iot',  'comm_ethernet',
                                       'comm_dsl', 'comm_arp',  'comm_hdlc',  'comm_lldp',  'comm_mac',  'comm_ppp',  'comm_stp', 'comm_vlan',  'comm_isis',
                                       'comm_mpls',  'comm_nat',  'comm_vrrp',  'comm_ip', 'comm_icmp',  'comm_rip',  'comm_ospf',  'comm_tcp',  'comm_udp',
                                       'comm_rpc',  'comm_ssl',  'comm_tls',  'comm_dhcp',  'comm_dns',  'comm_http',  'comm_snmp',  'comm_ftp',  'comm_ssh',
                                       'percentile','epss','kev_dateAdded','delta_fecha'],
                                       errors= 'ignore')
    Data_df = Data_df.dropna()

    y = Data_df["has_kev"].to_numpy(dtype = np.int8)
    X = Data_df.drop(columns = ["has_kev"]).to_numpy(dtype = np.float32)
    n_total = len(y)

    log(f"Dataset cargado: {X.shape}, positivos = {y.sum()}, negativos = {n_total - y.sum()}")

    # ========================================================================================
    # NOMBRE DE VARIABLES MODELOS Y RESULTADOS
    # ========================================================================================
    
    models = []
    confusion_matrices_test = []
    confusion_matrices_train = []

    errores = []

    ti = time.perf_counter()
    for i in range(num_exp):
        try:
            X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

            clf = LogisticRegression(
                solver=solver_v,
                C=c_v,
                penalty=penalty_v,
                class_weight="balanced",
                max_iter=1000
            )

            # =======================================
            # entrena y evalua el clasificador
            # =======================================

            y_pred = clf.fit(X_train,y_train).predict(X_test)
            y_pred_entrena = clf.predict(X_train)

            # =========================================
            # evaluacion del entrenamiento
            # =========================================
            conf_mat_train = confusion_matrix(y_train,y_pred_entrena)

            # =========================================
            # evaluacion de generalizacion
            # =========================================

            conf_mat_test = confusion_matrix(y_test,y_pred)

            models.append(clf)
            confusion_matrices_test.append(conf_mat_test)
            confusion_matrices_train.append(conf_mat_train)

            if (i+1)% checkpoint_every == 0:
                ckpt_path = f"pickle/checkpoints/ckpt_{combo_id}_{tag_df}_exp{i+1}.pckl"
                with open(ckpt_path,"wb") as f:
                    pickle.dump({
                        "solver": solver_v,
                        "C": c_v,
                        "penalty": penalty_v,
                        "modelos":models,
                        "conf_matrix_test":confusion_matrices_test,
                        "conf_matrix_train":confusion_matrices_train,
                        "errores": errores,
                    },f,protocol=pickle.HIGHEST_PROTOCOL)
                log(f"checkpoint guardado ({i+1} experimentos completados)")
        except Exception as e:
            msg = f"ERROR en exp = {i}: {type(e).__name__}: {e}"
            log(msg)
            errores.append(msg)
            continue 
    tf = time.perf_counter()
    del X,X_test,X_train, y, y_test,y_train
    gc.collect()

    log(f"TERMINADO combo. tiempo total: {tf-ti:.1f}s errores ={len(errores)} ")

    return {
        "solver": solver_v,
        "C": c_v,
        "penalty": penalty_v,
        "time_train": tf - ti,
        "modelos":models,
        "conf_matrix_test":confusion_matrices_test,
        "conf_matrix_train":confusion_matrices_train,
        "errores": errores
    }

# ==================================================
# LANZAR EN PARALELEO CON BARRA DE PROGRESO
# ==================================================

print(f"\nLanzando {len(param_list)} combinaciones en {num_cluster} engines...")
print(f"\nNUM_EXP ={num_exp}")
print("Revisa la carpeta pickle/checkpoints/ para ver avance real de cada engine.\n")

ar = dview.map_async(
    asignaciones,
    param_list,
    [num_exp] * len(param_list),
    [checkpoint_every] * len(param_list),
    [tag_df]*len(param_list),
    [tag_num_tex]*len(param_list)
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

all_models = []
all_conf_train = []
all_conf_test = []
for res in results:
    all_models.extend(res["modelos"])
    all_conf_train.extend(res["conf_matrix_train"])
    all_conf_test.extend(res["conf_matrix_test"])

with open(f"pickle/models_{tag_modelo}_{tag_df}_{tag_num_tex}.pckl", "wb") as f:
    pickle.dump(all_models, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"pickle/matrices_conf_train_{tag_modelo}_{tag_df}_{tag_num_tex}.pckl", "wb") as f:
    pickle.dump(all_conf_train, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(f"pickle/matrices_conf_test_{tag_modelo}_{tag_df}_{tag_num_tex}.pckl", "wb") as f:
    pickle.dump(all_conf_test, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"\nTotal modelos guardados: {len(all_models)}")
print("Listo.")