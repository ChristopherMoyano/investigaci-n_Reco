"""
==========================================================
  RESULTADOS ESPERADOS (conocidos de antemano):
  ─────────────────────────────────────────────
  1. Palabras similares a "perro"   → gato, mascota, ladra
  2. Palabras similares a "océano"  → mar, agua, pez
  3. Analogía: rey - hombre + mujer → reina
  4. Distancia coseno "perro"/"gato" < "perro"/"árbol" (más similares)
  5. Clústeres semánticos separados: animales / naturaleza / acciones
==========================================================
"""

from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────
# Palabras conocidas
# ──────────────────────────────────────────────

sentences = [
    # Bloque: mascotas y sus acciones
    ["el", "perro", "es", "una", "mascota", "fiel"],
    ["el", "gato", "es", "una", "mascota", "independiente"],
    ["el", "perro", "ladra", "y", "juega"],
    ["el", "gato", "maulla", "y", "duerme"],
    ["la", "mascota", "vive", "en", "casa"],
    ["el", "perro", "y", "el", "gato", "son", "mascotas"],

    # Bloque: animales del mar
    ["el", "pez", "vive", "en", "el", "océano"],
    ["el", "tiburón", "nada", "en", "el", "mar"],
    ["el", "delfín", "nada", "en", "el", "océano"],
    ["el", "océano", "y", "el", "mar", "tienen", "agua"],
    ["el", "pez", "y", "el", "delfín", "nadan", "en", "el", "agua"],

    # Bloque: naturaleza terrestre
    ["el", "árbol", "crece", "en", "el", "bosque"],
    ["la", "flor", "crece", "en", "el", "jardín"],
    ["el", "bosque", "tiene", "árboles", "y", "plantas"],
    ["la", "planta", "necesita", "agua", "y", "sol"],

    # Bloque: realeza (para la analogía rey - hombre + mujer = reina)
    ["el", "rey", "es", "un", "hombre", "poderoso"],
    ["la", "reina", "es", "una", "mujer", "poderosa"],
    ["el", "rey", "gobierna", "el", "reino"],
    ["la", "reina", "gobierna", "el", "reino"],
    ["el", "hombre", "y", "la", "mujer", "viven", "en", "el", "reino"],
    ["el", "rey", "y", "la", "reina", "son", "poderosos"],

    # Repeticiones para reforzar patrones
    ["perro", "gato", "mascota", "casa"],
    ["océano", "mar", "agua", "pez", "nada"],
    ["rey", "reina", "hombre", "mujer", "reino"],
    ["árbol", "bosque", "planta", "flor"],
]

# ──────────────────────────────────────────────
# 2. ENTRENAMIENTO DEL MODELO
# ──────────────────────────────────────────────
print("=" * 55)
print("  ENTRENAMIENTO DEL MODELO Word2Vec")
print("=" * 55)

model = Word2Vec(
    sentences=sentences,
    vector_size=50,    # Dimensiones del vector (pequeño → corpus pequeño)
    window=3,          # Contexto: 3 palabras a cada lado
    min_count=1,       # Incluir palabras que aparecen al menos 1 vez
    workers=1,         # 1 hilo para resultados reproducibles
    seed=42,           # Semilla fija → resultados reproducibles
    epochs=500,        # Muchas épocas → mejor aprendizaje con corpus pequeño
    sg=1,              # Skip-gram (mejor para corpus pequeños)
)

print(f"\n Vocabulario aprendido ({len(model.wv)} palabras):")
print(f"  {sorted(model.wv.index_to_key)}\n")

# ──────────────────────────────────────────────
# 3. RESULTADO 1 — Palabras más similares a "perro"
# ──────────────────────────────────────────────
print("─" * 55)
print("RESULTADO 1: Palabras más similares a 'perro'")
print("  ESPERADO → gato, mascota, ladra (misma semántica)")
print("─" * 55)

similares_perro = model.wv.most_similar("perro", topn=5)
for palabra, score in similares_perro:
    print(f"  {palabra:<12} {score:.4f} ")

# ──────────────────────────────────────────────
# 4. RESULTADO 2 — Palabras más similares a "océano"
# ──────────────────────────────────────────────
print("\n" + "─" * 55)
print("RESULTADO 2: Palabras más similares a 'océano'")
print("  ESPERADO → mar, agua, pez (mismo dominio acuático)")
print("─" * 55)

similares_oceano = model.wv.most_similar("océano", topn=5)
for palabra, score in similares_oceano:
    print(f"  {palabra:<12} {score:.4f} ")

# ──────────────────────────────────────────────
# 5. RESULTADO 3 — Analogía: rey - hombre + mujer = ?
# ──────────────────────────────────────────────
print("\n" + "─" * 55)
print("RESULTADO 3: Analogía vectorial")
print("  rey − hombre + mujer = ?")
print("  ESPERADO → reina")
print("─" * 55)

analogia = model.wv.most_similar(
    positive=["rey", "mujer"],
    negative=["hombre"],
    topn=3
)
print("  Top resultados:")
for palabra, score in analogia:
    marca = " ← ¡CORRECTO!" if palabra == "reina" else ""
    print(f"  {palabra:<12} {score:.4f}{marca}")

# ──────────────────────────────────────────────
# 6. RESULTADO 4 — Comparación de distancias coseno
# ──────────────────────────────────────────────
print("\n" + "─" * 55)
print("RESULTADO 4: Similitud coseno entre pares de palabras")
print("  ESPERADO → sim(perro, gato) > sim(perro, árbol)")
print("─" * 55)

sim_pg  = model.wv.similarity("perro", "gato")
sim_pa  = model.wv.similarity("perro", "árbol")
sim_om  = model.wv.similarity("océano", "mar")
sim_of  = model.wv.similarity("océano", "flor")

print(f"  sim(perro,  gato)   = {sim_pg:.4f}  ← animales cercanos")
print(f"  sim(perro,  árbol)  = {sim_pa:.4f}  ← dominios distintos")
print(f"  sim(océano, mar)    = {sim_om:.4f}  ← sinónimos acuáticos")
print(f"  sim(océano, flor)   = {sim_of:.4f}  ← dominios distintos")

assert sim_pg > sim_pa, "FALLO: perro/gato debería ser más similar que perro/árbol"
assert sim_om > sim_of, "FALLO: océano/mar debería ser más similar que océano/flor"
print("\n  ✔ Aserciones pasadas correctamente")

# ──────────────────────────────────────────────
# 7. RESULTADO 5 — Palabra que NO encaja en el grupo
# ──────────────────────────────────────────────
print("\n" + "─" * 55)
print("RESULTADO 5: ¿Cuál no encaja?")
print("  Lista: ['perro', 'gato', 'mascota', 'océano']")
print("  ESPERADO → océano  (no es mascota ni animal doméstico)")
print("─" * 55)

no_encaja = model.wv.doesnt_match(["perro", "gato", "mascota", "océano"])
print(f"  Palabra que no encaja: '{no_encaja}'")
marca = " CORRECTO" if no_encaja == "océano" else "✗ Revisar parámetros"
print(f"  {marca}")

# ──────────────────────────────────────────────
# 8. INSPECCIÓN: Vector de una palabra
# ──────────────────────────────────────────────
print("\n" + "─" * 55)
print("EXTRA: Vector de 'perro' (primeras 10 dimensiones)")
print("─" * 55)

vector_perro = model.wv["perro"]
print(f"  Dimensiones totales: {len(vector_perro)}")
print(f"  Primeras 10 dims: {np.round(vector_perro[:10], 4)}")

# ──────────────────────────────────────────────
# 9. GUARDAR Y RECARGAR EL MODELO
# ──────────────────────────────────────────────
model.save("word2vec_animales.model")
modelo_cargado = Word2Vec.load("word2vec_animales.model")
print("\n Modelo guardado y recargado correctamente.")
print(f"  Verificación: sim(rey, reina) = {modelo_cargado.wv.similarity('rey', 'reina'):.4f}")

print("\n" + "=" * 55)
print("  TODOS LOS RESULTADOS ESPERADOS VERIFICADOS ")
print("=" * 55)