import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import seaborn as sns
import scipy.stats as stats
import statistics
from math import sqrt
from datetime import datetime
import scipy.special as sp
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, recall_score, precision_score, roc_auc_score
import pandas as pd
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import RocCurveDisplay


from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

#============leer resultados============================
#vamos a usar clasificador tipo Decision Tree. Toca buscar alguno que sea representativo dentro de los 100

tag = "vuln_embeddings"

models = []
with open(f"pickle/models_tree_{tag}.pckl", "rb") as f:
    models = pickle.load(f) 
with open(f"pickle/matrices_conf_train_tree_{tag}.pckl", "rb") as f:
    confusion_matrices_train=pickle.load(f)
with open(f"pickle/matrices_conf_test_tree_{tag}.pckl", "rb") as f:
    confusion_matrices_test=pickle.load(f)         

# ===========================================================

# with open(f"pickle/tree_par_results_{tag}_100%.pckl", "rb") as f:
#     results = pickle.load(f)

# meta_rows = []

# for i, res in enumerate(results):
#     meta_rows.append({
#         "combo_id": i,
#         "splitter": res["splitter"],
#         "max_features": res["max_features"],
#         "max_depth": res["gamma"],
#         "time_train": res["time_train"],
#     })

#============Prediccion de vulnerabilidades explotadas============================
class_mean_acuracy_vector = []

num_exp=100


for j in range(0,16):
    MA_test = np.zeros((2,2))

    for i in range(j*num_exp,(j*num_exp)+num_exp):
        MA_test =MA_test + confusion_matrices_test[i]
    
    mean_recall_no_exp=MA_test[0][0]/(MA_test[0][0]+MA_test[0][1])
    
    mean_recall_exp=MA_test[1][1]/(MA_test[1][0]+MA_test[1][1])
    
    class_mean_acuracy=0.5*(mean_recall_no_exp + mean_recall_exp)
    
    class_mean_acuracy_vector.append(class_mean_acuracy)
    
    
max_CME = np.max(class_mean_acuracy_vector)  

position_max_CME = np.where(class_mean_acuracy_vector == max_CME)
# print(meta_rows[position_max_CME[0][0]])
position_max_CME[0][0] = position_max_CME[0][0]*100
print(models[position_max_CME[0][0]])

path1='dataset_vuln_embeddings.csv'
Data_df = pd.read_csv(path1)

Data_df = Data_df.drop(columns = ["version", "published", "date_reserved", "cwe_name", "published_year", "cwe_main", "ap_classOrName",
                                  "mo_phases", 'percentile', 'is_comm_system', 'epss','kev_dateAdded','delta_fecha'],errors='ignore') #hay que probar usar o no baseScore y epps, ya se usaran luego para evaluar

Data_df = Data_df.drop(columns = [ 'comm_router',  'comm_switch',  'comm_access_point',  'comm_firewall', 'comm_vpn',
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
 'comm_rpc',  'comm_ssl',  'comm_tls',  'comm_dhcp',  'comm_dns',  'comm_http',  'comm_snmp',  'comm_ftp',  'comm_ssh'],errors='ignore')

Data = Data_df.to_numpy()

ids = ["CVE-2023-42916", "CVE-2023-36851"]

# --- filtrar SOLO filas con has_kev = 1 ---
df_pos = Data_df[Data_df["has_kev"] == 1].copy()

# etiqueta y features (de SOLO positivos)
y = df_pos["has_kev"].to_numpy(dtype=np.int32)
X_df = df_pos.drop(columns=["has_kev"])
muestra_X_df = X_df.loc[X_df["cve_id"].isin(ids)].copy()
muestra_X_df.drop(columns=["cve_id"],inplace=True)

# tomar 2 filas al azar (SOLO de has_kev=1)
#muestra_X_df = X_df.sample(n=2, random_state=None)   # random_state=42 si quieres fijo
X = muestra_X_df.to_numpy(dtype=np.float32)


# clasificador (ojo con el índice si es combo vs modelo)
Clasif = models[position_max_CME[0][0]]

# predecir
Detector = Clasif.predict(X)

print("Índices elegidos:", muestra_X_df.index.to_list())
print("Predicción:", Detector)

# verificar real (debería ser [1 1])
y_real = Data_df.loc[muestra_X_df.index, "has_kev"].to_numpy(dtype=np.int32)
print("Real:", y_real)

idx = muestra_X_df.index
data = pd.read_csv(path1)

print(data.loc[idx]["cve_id"])

# #============Predicción de días explotados============================


# tag = "NVD"

# models = []
# with open(f"pickle/models_tree_{tag}_exp_7_days_100%.pckl", "rb") as f:
#     models = pickle.load(f) 
# with open(f"pickle/matrices_conf_train_tree_{tag}_exp_7_days_100%.pckl", "rb") as f:
#     confusion_matrices_train=pickle.load(f)
# with open(f"pickle/matrices_conf_test_tree_{tag}_exp_7_days_100%.pckl", "rb") as f:
#     confusion_matrices_test=pickle.load(f)         

# # # ===========================================================

# class_mean_acuracy_vector = []

# num_exp=100


# for j in range(0,16):
#     MA_test = np.zeros((2,2))

#     for i in range(j*num_exp,(j*num_exp)+num_exp):
#         MA_test =MA_test + confusion_matrices_test[i]
    
#     mean_recall_no_exp=MA_test[0][0]/(MA_test[0][0]+MA_test[0][1])
    
#     mean_recall_exp=MA_test[1][1]/(MA_test[1][0]+MA_test[1][1])
    
#     class_mean_acuracy=0.5*(mean_recall_no_exp + mean_recall_exp)
    
#     class_mean_acuracy_vector.append(class_mean_acuracy)
    
    
# max_CME = np.max(class_mean_acuracy_vector)  

# position_max_CME = np.where(class_mean_acuracy_vector == max_CME)
# position_max_CME[0][0] = position_max_CME[0][0]*100
# print(models[position_max_CME[0][0]])

# path1='csv/dataset_NVD_only_exp_7_days.csv'
# Data_df = pd.read_csv(path1)

# Data_df = Data_df.drop(columns = ["version", "published", "date_reserved", "cwe_name", "published_year", "cwe_main", "ap_classOrName",
#                                   "mo_phases", 'percentile', 'is_comm_system', 'epss','kev_dateAdded','delta_fecha'],errors='ignore') #hay que probar usar o no baseScore y epps, ya se usaran luego para evaluar

# Data_df = Data_df.drop(columns = [ 'comm_router',  'comm_switch',  'comm_access_point',  'comm_firewall', 'comm_vpn',
#  'comm_openvpn',  'comm_modem',  'comm_cisco_ios',  'comm_cisco_ios_xe',  'comm_cisco_nx_os',  'comm_junos',  'comm_routeros',
#  'comm_pan_os',  'comm_fortios',  'comm_sonicos',  'comm_checkpoint_gaia',  'comm_6wind',  'comm_barracuda',  'comm_f5',
#  'comm_kemp_technologies',  'comm_netapp',  'comm_netcracker',  'comm_pica8',  'comm_extremenetworks',  'comm_alcatel',
#  'comm_hpe', 'comm_aruba', 'comm_avm',  'comm_calix',  'comm_ericsson',  'comm_cisco',  'comm_d_link',  'comm_dlink',
#  'comm_comcast',  'comm_fiberhome',  'comm_fortinet',  'comm_huawei',  'comm_juniper',  'comm_mikrotik',  'comm_nec',
#  'comm_silver_peak',  'comm_tp_link',  'comm_zte',  'comm_cambium',  'comm_broadcom',  'comm_realtek',  'comm_arista',
#  'comm_linksys', 'comm_exinda',  'comm_infoblox',  'comm_lte',  'comm_5g',  'comm_gsm',  'comm_umts',  'comm_wifi',
#  'comm_bluetooth',  'comm_sip',  'comm_voip',  'comm_pbx',  'comm_sdn',  'comm_nfv',  'comm_iot',  'comm_ethernet',
#  'comm_dsl', 'comm_arp',  'comm_hdlc',  'comm_lldp',  'comm_mac',  'comm_ppp',  'comm_stp', 'comm_vlan',  'comm_isis',
#  'comm_mpls',  'comm_nat',  'comm_vrrp',  'comm_ip', 'comm_icmp',  'comm_rip',  'comm_ospf',  'comm_tcp',  'comm_udp',
#  'comm_rpc',  'comm_ssl',  'comm_tls',  'comm_dhcp',  'comm_dns',  'comm_http',  'comm_snmp',  'comm_ftp',  'comm_ssh'],errors='ignore')

# ids = ["CVE-2021-44228", "CVE-2023-20198"]

# # --- filtrar SOLO filas con has_kev = 1 ---
# df_pos = Data_df[Data_df["delta_menor_igual_7"] == 1].copy()

# # etiqueta y features (de SOLO positivos)
# y = df_pos["delta_menor_igual_7"].to_numpy(dtype=np.int32)
# X_df = df_pos.drop(columns=["delta_menor_igual_7"])
# muestra_X_df = X_df.loc[X_df["cve_id"].isin(ids)].copy()
# muestra_X_df.drop(columns=["cve_id"],inplace=True)

# # tomar 2 filas al azar (SOLO de has_kev=1)
# #muestra_X_df = X_df.sample(n=2, random_state=None)   # random_state=42 si quieres fijo
# X = muestra_X_df.to_numpy(dtype=np.float32)


# # clasificador (ojo con el índice si es combo vs modelo)
# Clasif = models[position_max_CME[0][0]]

# # predecir
# Detector = Clasif.predict(X)

# print("Índices elegidos:", muestra_X_df.index.to_list())
# print("Predicción:", Detector)

# # verificar real (debería ser [1 1])
# y_real = Data_df.loc[muestra_X_df.index, "delta_menor_igual_7"].to_numpy(dtype=np.int32)
# print("Real:", y_real)

# idx = muestra_X_df.index
# data = pd.read_csv(path1)


# print(data.loc[idx]["cve_id"])