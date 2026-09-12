import numpy as np
import ipyparallel as ipp
from timeit import default_timer as timer
from itertools import product
import pandas as pd
import pickle



start = timer ()


num_cluster = 12

cluster = ipp.Cluster(n = num_cluster)
cluster.start_cluster_sync()

#conectar un cliente con los clusters
rc = cluster.connect_client_sync()

print(rc.ids)

rc.wait_for_engines(num_cluster)

dview = rc[:]

splitter_arr = ["best","random"]
max_features_arr = [None, "sqrt"]
max_depth_arr = [None,1500]
min_samples_leaf_arr = [1,1100]
param_list = list(product(splitter_arr, max_features_arr, max_depth_arr,min_samples_leaf_arr))
tag = "vuln_embeddings"
tag2="texto+num"




def asignaciones(params):
    import numpy as np
    from sklearn import svm
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import time
    from sklearn.metrics import confusion_matrix
    from sklearn import tree
    from imblearn.under_sampling import RandomUnderSampler
    import gc

    splitter_v, max_features_v, max_depth_v,min_samples_leaf_v = params
    tag = "vuln_embeddings"

    path1='dataset_'+tag+'.csv'

    Data_df = pd.read_csv(path1)#, usecols=[1,2,3])
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

    # #============nombre variables modelos y resultados=====================
    models = []
    confusion_matrices_test=[]
    confusion_matrices_train=[]

    num_exp=50
    num_under=30

    t0 = time.perf_counter()
    for j in range (0,num_under):
        y = Data_df["has_kev"] #.to_numpy(dtype=np.int8)
        X = Data_df.drop(columns=["has_kev"]) #.to_numpy(dtype=np.float32)


        rus = RandomUnderSampler(sampling_strategy=1)
        X_res, y_res = rus.fit_resample(X,y)


        indices_a_borrar = X_res.index
        X_otro = X.drop(indices_a_borrar).to_numpy(dtype=np.float32)
        y_otro = y.drop(indices_a_borrar).to_numpy(dtype=np.int8)

        y_res = y_res.to_numpy(dtype=np.int8)
        X_res = X_res.to_numpy(dtype=np.float32)

        
    
        for i in range(0,num_exp):

            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3)

            X_test = np.vstack([X_test,X_otro])  
            y_test = np.concatenate([y_test,y_otro])


        # ---------- entrenar SVM ----------

            clf = tree.DecisionTreeClassifier(
                            splitter=splitter_v,
                            class_weight="balanced",
                            max_depth=max_depth_v,
                            min_samples_leaf=min_samples_leaf_v,
                            max_features=max_features_v
                        )

            #entrenar y evaluar el clasif 
            y_pred = clf.fit(X_train, y_train).predict(X_test)

            y_pred_entrena = clf.predict(X_train)


            # ---------------evaluacion del entrenamiento-------------------
            conf_mat_train=confusion_matrix(y_train, y_pred_entrena)
            # ---------------evaluacion de generalizacion-------------------
            conf_mat_test=confusion_matrix(y_test, y_pred)


            models.append(clf)
            confusion_matrices_test.append(conf_mat_test)
            confusion_matrices_train.append(conf_mat_train)

        del X_res, y_res, X_otro, y_otro
        gc.collect()   
    
    # devolvemos un diccionario con resultados
    
    t1 = time.perf_counter()
    
    return {
        "splitter": splitter_v,
        "max_features": max_features_v,
        "max_depth": max_depth_v,
        "min_samples_leaf":min_samples_leaf_v,
        "time_train": t1 - t0,
        "modelos":models,
        "conf_matrix_test":confusion_matrices_test,
        "conf_matrix_train":confusion_matrices_train
    }

    
ar = dview.map_async(asignaciones, param_list)


results = ar.get()

# ============ almacenar TODOS los resultados juntos ===============
with open(f"pickle/tree_par_results_{tag}_{tag2}.pckl", "wb") as f:
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

# # ============ leer TODOS los resultados juntos ===============
# with open("svm_par_results_v1.pckl", "rb") as f:
#     results = pickle.load(f)

# meta_rows = []

# for i, res in enumerate(results):
#     meta_rows.append({
#         "combo_id": i,
#         "kernel": res["kernel"],
#         "C": res["C"],
#         "gamma": res["gamma"],
#         "time_train": res["time_train"],
#     })

# meta_df = pd.DataFrame(meta_rows)
# print(meta_df.info())

# all_models = []
# all_conf_train = []
# all_conf_test = []

# for res in results:
#     all_models.extend(res["modelos"])
#     all_conf_train.extend(res["conf_matrix_train"])
#     all_conf_test.extend(res["conf_matrix_test"])

# print("Total modelos:", len(all_models))
# print("Total matrices train:", len(all_conf_train))
# print("Total matrices test:", len(all_conf_test))

# #ver la matriz de confusión del modelo global 0
# print("Conf train [0]:\n", all_conf_train[0])
# print("Conf test  [0]:\n", all_conf_test[0])




# ========= aplanar resultados desde 'results' ==========
all_models = []
all_conf_train = []
all_conf_test = []


for res in results:
    # res es el diccionario devuelto por asignaciones
    all_models.extend(res["modelos"])                # agrega los 100 modelos
    all_conf_train.extend(res["conf_matrix_train"])  # agrega 100 matrices train
    all_conf_test.extend(res["conf_matrix_test"])    # agrega 100 matrices test


# =======================================================

# ============= almacenar resultados ======================
with open(f"pickle/models_tree_{tag}_{tag2}.pckl", "wb") as f:
    pickle.dump(all_models, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(f"pickle/matrices_conf_train_tree_{tag}_{tag2}.pckl", "wb") as f:
    pickle.dump(all_conf_train, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(f"pickle/matrices_conf_test_tree_{tag}_{tag2}.pckl", "wb") as f:
    pickle.dump(all_conf_test, f, protocol=pickle.HIGHEST_PROTOCOL)
# =========================================================

# # ============= leer resultados ===========================
# with open("models02_v3.pckl", "rb") as f:
#     all_models = pickle.load(f)

# with open("matrices_conf_train02_v3.pckl", "rb") as f:
#     all_conf_train = pickle.load(f)

# with open("matrices_conf_test02_v3.pckl", "rb") as f:
#     all_conf_test = pickle.load(f)
# # =========================================================