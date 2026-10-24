# ==========================================================
# QuantumFire Perú — Dashboard de Optimización Híbrida (monolito Streamlit)
# ==========================================================
# Ejecutar desde la raíz del repo (para que cargue .streamlit/config.toml):
#   pip install -r requirements.txt
#   streamlit run src/quantumfire_app.py
#
# Usa src/servicio.py (asignar_brigadas) como motor de cálculo.

import json

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import servicio as srv   # streamlit añade la carpeta del script al sys.path

st.set_page_config(page_title="QuantumFire Perú — Dashboard de Optimización Híbrida",
                   page_icon="🔥", layout="wide")

AZUL, NARANJA, FUEGO, GRIS, FONDO = "#2F80ED", "#FF9F1C", "#FF3B1F", "#5B6785", "#0B1020"
VEGETACIONES = ["bosque_denso", "pastizal", "matorral"]

st.markdown("""
<style>
.qf-titulo {font-size: 2.1rem; font-weight: 800; margin-bottom: 0;
            background: linear-gradient(90deg, #2F80ED, #FF5A1F); -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;}
.qf-sub {color: #9FB0D9; margin-top: 0;}
</style>
""", unsafe_allow_html=True)


# ---------- Datos por defecto (sintéticos) ----------

def df_bases_defecto():
    return pd.DataFrame([
        {"id": "Base_Cusco",   "lat": -13.5319, "lon": -71.9675, "helipuerto": True},
        {"id": "Base_Abancay", "lat": -13.6339, "lon": -72.8814, "helipuerto": False},
    ])


def df_zonas_defecto():
    return pd.DataFrame([
        {"id": "Zona_Calca",       "lat": -13.3200, "lon": -71.9600, "vegetacion": "pastizal",
         "acceso_terrestre": True,  "viento_kmh": 25.0, "area_ha": 20.0, "intensidad": 0.8, "propagacion_ha_h": 10.0},
        {"id": "Zona_Paucartambo", "lat": -13.3200, "lon": -71.5900, "vegetacion": "bosque_denso",
         "acceso_terrestre": False, "viento_kmh": 40.0, "area_ha": 35.0, "intensidad": 0.6, "propagacion_ha_h": 6.0},
    ])


def df_recursos_defecto():
    # Mismos parámetros que el servicio + velocidad_kmh (para convertir distancia en minutos de tránsito)
    return pd.DataFrame([
        {"id": "Helicoptero", "costo_operativo": 4.0, "capacidad_mitigacion": 0.9, "tiempo_preparacion": 5.0,
         "capacidad_ha_h": 12.0, "viento_max_kmh": 50.0, "requiere_acceso_terrestre": False,
         "velocidad_kmh": 200.0, "apt_bosque_denso": 0.9, "apt_pastizal": 1.0, "apt_matorral": 0.9},
        {"id": "CisternaPesada", "costo_operativo": 2.0, "capacidad_mitigacion": 0.6, "tiempo_preparacion": 10.0,
         "capacidad_ha_h": 5.0, "viento_max_kmh": 999.0, "requiere_acceso_terrestre": True,
         "velocidad_kmh": 40.0, "apt_bosque_denso": 0.5, "apt_pastizal": 1.0, "apt_matorral": 0.8},
    ])


def df_brigadas_defecto():
    return pd.DataFrame([
        {"brigada": "B1", "base": "Base_Cusco",   "transporte": "Helicoptero"},
        {"brigada": "B2", "base": "Base_Abancay", "transporte": "CisternaPesada"},
    ])


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# ---------- Estado de sesión ----------

for k, f in (("src_bases", df_bases_defecto), ("src_zonas", df_zonas_defecto),
             ("src_recursos", df_recursos_defecto), ("src_brigadas", df_brigadas_defecto)):
    if k not in st.session_state:
        st.session_state[k] = f()
st.session_state.setdefault("clic", None)
st.session_state.setdefault("resultado", None)


def agregar_punto(tipo, nombre, lat, lon):
    """Callback del botón del mapa: añade el punto y reinicia el editor para que tome la nueva fila."""
    if tipo == "Zona de incendio":
        base = st.session_state.get("zonas_actual", st.session_state.src_zonas)
        fila = {"id": nombre, "lat": lat, "lon": lon, "vegetacion": "pastizal", "acceso_terrestre": True,
                "viento_kmh": 20.0, "area_ha": 10.0, "intensidad": 0.6, "propagacion_ha_h": 5.0}
        st.session_state.src_zonas = pd.concat([base, pd.DataFrame([fila])], ignore_index=True)
        st.session_state.pop("ed_zonas", None)
    else:
        base = st.session_state.get("bases_actual", st.session_state.src_bases)
        fila = {"id": nombre, "lat": lat, "lon": lon, "helipuerto": False}
        st.session_state.src_bases = pd.concat([base, pd.DataFrame([fila])], ignore_index=True)
        st.session_state.pop("ed_bases", None)
    st.session_state.clic = None


# ---------- Construcción del escenario ----------

def construir_escenario(bases, zonas, recursos, brigadas, prioridades):
    """Convierte lo ingresado en el dashboard a los parámetros de asignar_brigadas."""
    ids_b, ids_z, ids_r = list(bases["id"]), list(zonas["id"]), list(recursos["id"])
    for nombre, ids in (("bases", ids_b), ("zonas", ids_z), ("recursos", ids_r)):
        if not ids or any(not str(i).strip() for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"Los ids de {nombre} deben ser no vacíos y únicos (y haber al menos uno).")
    if brigadas.empty:
        raise ValueError("Debe haber al menos una brigada.")
    if not brigadas["base"].isin(ids_b).all() or not brigadas["transporte"].isin(ids_r).all():
        raise ValueError("Alguna brigada usa una base o transporte que ya no existe en las tablas.")
    if recursos[["velocidad_kmh"]].le(0).any().any():
        raise ValueError("velocidad_kmh debe ser > 0.")
    if bases[["lat", "lon"]].isna().any().any() or zonas.isna().any().any() or recursos.isna().any().any():
        raise ValueError("Hay celdas vacías en las tablas de bases, zonas o recursos.")

    vel = recursos.set_index("id")["velocidad_kmh"]
    # minutos de tránsito [base][zona][recurso]
    matriz = np.array([[[haversine_km(b.lat, b.lon, z.lat, z.lon) / vel[r] * 60.0 for r in ids_r]
                        for z in zonas.itertuples()] for b in bases.itertuples()])

    props_recursos = {}
    for r in recursos.itertuples():
        props_recursos[r.id] = {
            "costo_operativo": float(r.costo_operativo), "capacidad_mitigacion": float(r.capacidad_mitigacion),
            "tiempo_preparacion": float(r.tiempo_preparacion), "capacidad_ha_h": float(r.capacidad_ha_h),
            "viento_max_kmh": float(r.viento_max_kmh),
            "requiere_acceso_terrestre": bool(r.requiere_acceso_terrestre),
            "aptitud_vegetacion": {"bosque_denso": float(r.apt_bosque_denso), "pastizal": float(r.apt_pastizal),
                                   "matorral": float(r.apt_matorral)}}

    disponibilidad = {b: {r: 0 for r in ids_r} for b in ids_b}
    for br in brigadas.itertuples():
        disponibilidad[br.base][br.transporte] += 1

    return dict(
        bases=ids_b, zonas=ids_z, recursos=ids_r,
        riesgo_zonas={z: prioridades[z] / 5.0 for z in ids_z},     # p_j in [1,5] -> riesgo in [0.2,1]
        props_recursos=props_recursos,
        infraestructura_bases={b.id: {"tiene_helipuerto": bool(b.helipuerto)} for b in bases.itertuples()},
        stock_disponible={r: int((brigadas["transporte"] == r).sum()) for r in ids_r},
        matriz_transito=matriz,
        props_zonas={z.id: {"vegetacion": z.vegetacion, "acceso_terrestre": bool(z.acceso_terrestre),
                            "viento_kmh": float(z.viento_kmh)} for z in zonas.itertuples()},
        fuego_zonas={z.id: {"area_ha": float(z.area_ha), "intensidad": float(z.intensidad),
                            "propagacion_ha_h": float(z.propagacion_ha_h)} for z in zonas.itertuples()},
        capacidad_base={b: int((brigadas["base"] == b).sum()) for b in ids_b},
        disponibilidad=disponibilidad)


def matriz_cij(esc, brigadas):
    """c_ij en minutos de tránsito: fila = brigada (su base y transporte), columna = zona."""
    ib = {b: i for i, b in enumerate(esc["bases"])}
    ir = {r: k for k, r in enumerate(esc["recursos"])}
    datos = [[esc["matriz_transito"][ib[br.base], j, ir[br.transporte]] for j in range(len(esc["zonas"]))]
             for br in brigadas.itertuples()]
    return pd.DataFrame(datos, index=list(brigadas["brigada"]), columns=esc["zonas"])


def asignar_ids_brigada(asignaciones, brigadas, cij):
    """Asigna una brigada concreta (id) a cada (base, zona, recurso) devuelto por el servicio."""
    libres = brigadas.copy()
    filas = []
    for a in asignaciones:
        cand = libres[(libres["base"] == a["base"]) & (libres["transporte"] == a["recurso"])]
        if cand.empty:
            continue
        idb = cand.iloc[0]["brigada"]
        libres = libres[libres["brigada"] != idb]
        filas.append({"brigada": idb, "base": a["base"], "zona": a["zona"], "recurso": a["recurso"],
                      "transito_min": round(float(cij.loc[idb, a["zona"]]), 1)})
    return pd.DataFrame(filas)


# ---------- Gráfico ----------

def dibujar_grafo(brigadas, cij, esc_res, asignadas):
    """Grafo bipartito: brigadas (azul, izquierda) -> zonas (naranja, derecha), aristas con c_ij en min."""
    factibles = {(o["base"], o["zona"], o["recurso"]) for o in esc_res["opciones_factibles"]} if esc_res else None
    G = nx.Graph()
    brig = list(brigadas.itertuples())
    zonas = list(cij.columns)
    pos = {}
    for i, b in enumerate(brig):
        G.add_node(b.brigada)
        pos[b.brigada] = (0, -i)
    for j, z in enumerate(zonas):
        G.add_node(z)
        pos[z] = (1, -j * max(len(brig) - 1, 1) / max(len(zonas) - 1, 1))

    asig = {(r.brigada, r.zona) for r in asignadas.itertuples()} if asignadas is not None and len(asignadas) else set()
    e_asig, e_fact, e_inf, etiquetas = [], [], [], {}
    for b in brig:
        for z in zonas:
            ok = factibles is None or (b.base, z, b.transporte) in factibles
            G.add_edge(b.brigada, z)
            if (b.brigada, z) in asig:
                e_asig.append((b.brigada, z))
            elif ok:
                e_fact.append((b.brigada, z))
            else:
                e_inf.append((b.brigada, z))
            if ok:
                etiquetas[(b.brigada, z)] = f"{cij.loc[b.brigada, z]:.0f} min"

    fig, ax = plt.subplots(figsize=(9, max(3.5, 1.1 * len(brig) + 1)))
    fig.patch.set_facecolor(FONDO)
    ax.set_facecolor(FONDO)
    nx.draw_networkx_edges(G, pos, edgelist=e_inf, edge_color=GRIS, style="dashed", width=0.8, alpha=0.4, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=e_fact, edge_color="#6C7FB0", width=1.2, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=e_asig, edge_color=FUEGO, width=4, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[b.brigada for b in brig], node_color=AZUL, node_size=1500, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=zonas, node_color=NARANJA, node_size=1900, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color="white", font_size=8, font_weight="bold", ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=etiquetas, label_pos=0.3, font_size=8, font_color="#E6ECFF",
                                 bbox=dict(facecolor=FONDO, edgecolor="none", pad=0.5), ax=ax)
    ax.set_xlim(-0.25, 1.25)
    ax.axis("off")
    ax.set_title("Brigadas (azul)  →  Zonas de incendio (naranja)   |   rojo = asignación óptima",
                 color="#9FB0D9", fontsize=10)
    return fig


# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.header("⚙️ Panel de inputs")

    # a) Selector de ubicación en el mapa
    with st.expander("📍 a) Mapa: ubicar zona afectada / base", expanded=True):
        bases_prev = st.session_state.get("bases_actual", st.session_state.src_bases)
        zonas_prev = st.session_state.get("zonas_actual", st.session_state.src_zonas)
        centro = [float(np.nanmean(list(bases_prev["lat"]) + list(zonas_prev["lat"]))),
                  float(np.nanmean(list(bases_prev["lon"]) + list(zonas_prev["lon"])))]
        m = folium.Map(location=centro, zoom_start=8, tiles="OpenStreetMap", control_scale=True)
        folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                         attr="Esri World Imagery", name="Satélite").add_to(m)
        for b in bases_prev.itertuples():
            folium.Marker([b.lat, b.lon], tooltip=f"Base: {b.id}", icon=folium.Icon(color="blue", icon="home")).add_to(m)
        for z in zonas_prev.itertuples():
            folium.Marker([z.lat, z.lon], tooltip=f"Zona: {z.id}", icon=folium.Icon(color="orange", icon="fire")).add_to(m)
        if st.session_state.clic:
            folium.Marker([st.session_state.clic["lat"], st.session_state.clic["lng"]], tooltip="Punto seleccionado",
                          icon=folium.Icon(color="green", icon="screenshot")).add_to(m)
        folium.LayerControl().add_to(m)
        salida_mapa = st_folium(m, height=300, width=290, returned_objects=["last_clicked"], key="mapa")
        if salida_mapa and salida_mapa.get("last_clicked") and salida_mapa["last_clicked"] != st.session_state.clic:
            st.session_state.clic = salida_mapa["last_clicked"]
            st.rerun()
        st.caption("Haz clic en el mapa para colocar el puntero. (Mapa abierto OSM/Esri: Google Maps exige API key.)")
        if st.session_state.clic:
            lat, lon = st.session_state.clic["lat"], st.session_state.clic["lng"]
            st.write(f"Puntero: `{lat:.4f}, {lon:.4f}`")
            tipo = st.radio("Agregar como", ["Zona de incendio", "Base"], horizontal=True)
            nombre = st.text_input("Id / nombre", value=f"{'Zona' if tipo.startswith('Zona') else 'Base'}_{len(zonas_prev) + len(bases_prev) + 1}")
            st.button("➕ Agregar punto", on_click=agregar_punto, args=(tipo, nombre, lat, lon), width="stretch")

    # Bases
    with st.expander("🏢 Bases", expanded=False):
        st.caption("Ubicación, helipuerto. (Filas editables; ➕/🗑️ para agregar o quitar.)")
        bases = st.data_editor(st.session_state.src_bases, num_rows="dynamic", key="ed_bases", width="stretch",
                               column_config={"helipuerto": st.column_config.CheckboxColumn("helipuerto")})
        st.session_state.bases_actual = bases

    # Recursos / tipos de transporte
    with st.expander("🚁 Tipos de transporte y sus características", expanded=False):
        st.caption("Mismos parámetros del servicio + velocidad (convierte distancia en minutos de tránsito).")
        recursos = st.data_editor(st.session_state.src_recursos, key="ed_recursos", width="stretch",
                                  disabled=["id"],
                                  column_config={"requiere_acceso_terrestre": st.column_config.CheckboxColumn("acceso terr.")})

    # b), c), d) Brigadas
    st.subheader("👥 Brigadas")
    n_brig = st.number_input("b) Cantidad de brigadas", min_value=1, max_value=12,
                             value=len(st.session_state.src_brigadas), step=1)
    if n_brig != len(st.session_state.src_brigadas):
        actual = st.session_state.get("brigadas_actual", st.session_state.src_brigadas)
        if n_brig < len(actual):
            nuevo = actual.iloc[:n_brig]
        else:
            extra = [{"brigada": f"B{i + 1}", "base": str(bases["id"].iloc[0]) if len(bases) else "",
                      "transporte": str(recursos["id"].iloc[0])} for i in range(len(actual), n_brig)]
            nuevo = pd.concat([actual, pd.DataFrame(extra)], ignore_index=True)
        st.session_state.src_brigadas = nuevo.reset_index(drop=True)
        st.session_state.pop("ed_brigadas", None)
    brigadas = st.data_editor(
        st.session_state.src_brigadas, key="ed_brigadas", width="stretch", disabled=["brigada"],
        column_config={
            "base": st.column_config.SelectboxColumn("c) Base", options=[str(i) for i in bases["id"]], required=True),
            "transporte": st.column_config.SelectboxColumn("d) Transporte", options=[str(i) for i in recursos["id"]],
                                                           required=True)})
    st.session_state.brigadas_actual = brigadas

    # Zonas
    with st.expander("🔥 Zonas de incendio: características y fuego", expanded=False):
        zonas = st.data_editor(st.session_state.src_zonas, num_rows="dynamic", key="ed_zonas", width="stretch",
                               column_config={
                                   "vegetacion": st.column_config.SelectboxColumn("vegetación", options=VEGETACIONES),
                                   "acceso_terrestre": st.column_config.CheckboxColumn("acceso terr.")})
        st.session_state.zonas_actual = zonas

    st.subheader("🎚️ Prioridad / riesgo por zona")
    prioridades = {}
    for z in zonas["id"]:
        if str(z).strip():
            prioridades[z] = st.slider(f"{z}  ($p_j$)", 1, 5, 3, key=f"prio_{z}")

    st.subheader("🧩 Restricciones y solver")
    dura = st.toggle("Restricción dura: Máximo 1 brigada por zona y 1 zona por brigada", value=False,
                     help="1 zona por brigada siempre se cumple (cada brigada se envía a una sola zona). "
                          "Activado, además, cada zona recibe como máximo 1 brigada.")
    cobertura = st.toggle("Exigir cobertura de toda zona alcanzable", value=True,
                          help="Si hay menos brigadas que zonas, el problema es infactible: desactívalo para priorizar.")
    with st.expander("Pesos de la función objetivo"):
        w_costo = st.slider("Costo", 0.0, 1.0, 0.25, 0.05)
        w_tiempo = st.slider("Tiempo", 0.0, 1.0, 0.25, 0.05)
        w_benef = st.slider("Beneficio", 0.0, 2.0, 1.0, 0.1)
        w_sin = st.slider("Sinergia aire+tierra", 0.0, 1.0, 0.3, 0.05)
    with st.expander("Parámetros QAOA"):
        usar_qaoa = st.checkbox("Ejecutar QAOA (si no, solo solver exacto)", value=True)
        reps = st.number_input("Capas (reps, p)", 1, 5, 1)
        maxiter = st.number_input("Iteraciones COBYLA", 10, 500, 100, step=10)
        max_qubits = st.slider("Máx. qubits para simular QAOA", 6, 24, 16)

    ejecutar = st.button("🚀 EJECUTAR OPTIMIZACIÓN", type="primary", width="stretch")

# ==========================================================
# EJECUCIÓN
# ==========================================================
if ejecutar:
    try:
        esc = construir_escenario(bases, zonas, recursos, brigadas, prioridades)
        with st.spinner("Resolviendo QUBO (QAOA simulado + referencia exacta)..."):
            res = srv.asignar_brigadas(
                **esc, pesos={"costo": w_costo, "tiempo": w_tiempo, "beneficio": w_benef, "sinergia": w_sin},
                usar_qaoa=usar_qaoa, reps=int(reps), maxiter=int(maxiter), max_qubits=int(max_qubits),
                max_una_por_zona=dura, cobertura_obligatoria=cobertura)
        cij = matriz_cij(esc, brigadas)
        st.session_state.resultado = {
            "res": res, "cij": cij, "brigadas": brigadas.copy(), "prioridades": dict(prioridades),
            "zonas": zonas.copy(), "asignadas": asignar_ids_brigada(res["asignaciones"], brigadas, cij)}
    except ValueError as e:
        st.session_state.resultado = None
        st.error(f"Entrada inválida: {e}")

R = st.session_state.resultado

# ==========================================================
# ÁREA PRINCIPAL
# ==========================================================
st.markdown('<p class="qf-titulo">🔥 QuantumFire Perú — Dashboard de Optimización Híbrida</p>', unsafe_allow_html=True)
st.markdown('<p class="qf-sub">Asignación de brigadas a incendios forestales con QAOA (Qiskit) · Qiskit Fall Fest 2026</p>',
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📍 Escenario y grafo de conexión", "📊 Transparencia y límites tecnológicos (NISQ)",
                            "🎯 Resultado de la optimización"])

# Vista previa del escenario aun sin ejecutar
try:
    esc_vista = construir_escenario(bases, zonas, recursos, brigadas,
                                    {z: prioridades.get(z, 3) for z in zonas["id"]})
    cij_vista = matriz_cij(esc_vista, brigadas)
    err_vista = None
except ValueError as e:
    esc_vista = cij_vista = None
    err_vista = str(e)

with tab1:
    st.markdown("""
**El problema (Facility Location / asignación):** dado un conjunto de brigadas ubicadas en bases y un conjunto de
zonas en llamas, decidir **qué brigada atiende qué zona** minimizando costos y tiempos de respuesta y maximizando la
mitigación esperada, ponderada por el riesgo $p_j$ de cada zona. Cada decisión es una variable binaria
$x_{b,z,r}$ (base $b$ envía el recurso $r$ a la zona $z$), lo que da un problema **QUBO** apto para QAOA.
""")
    if err_vista:
        st.error(f"Escenario incompleto: {err_vista}")
    else:
        if R and R["res"]["opciones_factibles"] is not None:
            fig = dibujar_grafo(R["brigadas"], R["cij"], R["res"], R["asignadas"])
            cij_mostrar, prios = R["cij"], R["prioridades"]
        else:
            fig = dibujar_grafo(brigadas, cij_vista, None, None)
            cij_mostrar, prios = cij_vista, prioridades
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.caption("Líneas punteadas: combinaciones imposibles (helipuerto, viento, acceso) — solo se marcan tras ejecutar. "
                   "Etiquetas: tiempo de tránsito $c_{ij}$ en minutos (distancia geodésica / velocidad del transporte).")
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**Matriz de costos $c_{ij}$ — tiempo de viaje (min)**")
            st.dataframe(cij_mostrar.round(1), width="stretch")
        with c2:
            st.markdown("**Prioridad de las zonas $p_j$**")
            st.dataframe(pd.DataFrame({"zona": list(prios), "p_j (1-5)": list(prios.values()),
                                       "riesgo (=p_j/5)": [v / 5 for v in prios.values()]}),
                         hide_index=True, width="stretch")

with tab2:
    st.subheader("Procedencia de los datos")
    st.warning("**Datos sintéticos.** Las bases, zonas, brigadas, parámetros de recursos, áreas de fuego, vientos y "
               "prioridades de este dashboard son ilustrativos. **No se ha integrado ningún dato real verificado.**")
    st.info("**Fuentes reales previstas (no conectadas):** SERFOR (cobertura forestal y focos), INDECI (emergencias y "
            "recursos), SENAMHI (viento y meteorología). Antes de usar el sistema operativamente, los parámetros deben "
            "calibrarse con esas fuentes y con las brigadas reales.")
    st.subheader("Límites de la computación cuántica actual (era NISQ)")
    st.warning("**Qué resuelve:** la **optimización de asignación espacial** (formulada como QUBO). "
               "**Qué NO resuelve:** la **simulación termodinámica/física del fuego** (propagación, combustión, "
               "convección), que requiere GPUs o supercomputadoras clásicas. Aquí la propagación es solo un "
               "parámetro de entrada (ha/h).")
    st.info("**Sobre QAOA aquí:** se ejecuta en un **simulador clásico** (statevector), no en hardware cuántico real. "
            "Cada variable y cada holgura de las restricciones cuesta un qubit, por lo que el tamaño simulable está limitado "
            "(slider de qubits máximos). A este tamaño el solver clásico exacto resuelve el problema al instante: se "
            "incluye como referencia para validar a QAOA, **no se afirma ventaja cuántica**.")
    st.subheader("Cómo se traducen los inputs al modelo")
    st.markdown("""
| Input del dashboard | Parámetro del servicio |
|---|---|
| Ubicación de bases y zonas (mapa) | Distancia geodésica → `matriz_transito` (min) con la velocidad del transporte |
| Cantidad de brigadas, base y tipo de transporte | `disponibilidad[base][recurso]` (máx. envíos por base y tipo) |
| Prioridad $p_j \\in [1,5]$ | `riesgo_zonas = p_j / 5` |
| Vegetación, viento, acceso, área, intensidad, propagación | `props_zonas`, `fuego_zonas` |
| Características del transporte | `props_recursos` (costo, mitigación, ha/h, viento máx., preparación, aptitud) |
| Restricción dura | `max_una_por_zona` |
""")
    st.latex(r"\min \sum_{b,z,r} \Big(w_c\,\tilde c_r + w_t\,\tilde t_{bzr} - w_b\,\beta_{bzr}\Big)x_{bzr} "
             r"- w_s \sum_{z}\sum_{r\neq r'} x_{bzr}\,x_{b'zr'}")

with tab3:
    if not R:
        st.info("Configura el escenario en la barra lateral y pulsa **🚀 EJECUTAR OPTIMIZACIÓN**.")
    else:
        res = R["res"]
        if res["estado"] == "OK":
            st.success(f"Asignación encontrada con **{res['metodo'].upper()}**.")
        elif res["estado"] == "SIN_VARIABLES":
            st.error("Ninguna combinación base/zona/recurso es factible (helipuerto, viento, acceso o brigadas).")
        else:
            st.error(f"Problema {res['estado']}: con las restricciones actuales no hay asignación válida "
                     "(prueba desactivando la cobertura obligatoria o agregando brigadas).")
        for z in res["zonas_sin_cobertura"]:
            st.warning(f"**{z}** no puede ser atendida por ninguna brigada (viento, acceso terrestre o helipuerto).")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Estado", res["estado"])
        m2.metric("Objetivo (menor = mejor)", "—" if res["objetivo"] is None else f"{res['objetivo']:.4f}")
        m3.metric("Qubits del QUBO", res["qubits"] if res["qubits"] is not None else "—")
        coincide = res["qaoa_coincide_con_exacto"]
        m4.metric("QAOA = exacto", "—" if coincide is None else ("Sí ✅" if coincide else "No ⚠️"))
        if res["qubits"] and usar_qaoa and res["qaoa"] is None:
            st.info(f"QAOA omitido: {res['qubits']} qubits superan el máximo configurado. Resultado del solver exacto.")

        if len(R["asignadas"]):
            st.markdown("**Asignación (ids para integración)**")
            st.dataframe(R["asignadas"], hide_index=True, width="stretch")
        st.markdown("**Respuesta JSON del servicio**")
        salida_json = {k: v for k, v in res.items()}
        salida_json["asignaciones_brigada"] = R["asignadas"].to_dict(orient="records")
        st.json(salida_json, expanded=False)
        st.download_button("⬇️ Descargar JSON", json.dumps(salida_json, indent=2, ensure_ascii=False),
                           file_name="asignacion.json", mime="application/json", width="stretch")
