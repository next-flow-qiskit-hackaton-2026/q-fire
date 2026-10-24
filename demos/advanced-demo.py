# ==========================================================
# DEMO AVANZADA: ASIGNACIÓN DE BRIGADAS A ZONAS DE INCENDIO (QAOA)
# ==========================================================
# Colab:  !pip install qiskit qiskit-optimization qiskit-algorithms
#
# Variables binarias x[base, zona, recurso] = 1 si el recurso sale de la base hacia la zona.
# Se agregan sobre los datos base (generar_datos_sistema_emergencia):
#   - características de cada brigada/recurso (velocidad de respuesta, ha/h, aptitud por vegetación, límites de viento)
#   - características de cada zona (vegetación, acceso terrestre, viento)
#   - distribución del fuego (área, intensidad, velocidad de propagación)
# Para sumar más variables basta con editar extender_datos() y/o coeficiente_lineal().

import itertools
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA


# ---------- 1. DATOS ----------

def generar_datos_sistema_emergencia():
    """Copia de demos/generate-data.py (con el import de numpy que le faltaba)."""
    bases = ["Base_Norte", "Base_Sur"]
    zonas = ["Zona_Critica_1", "Zona_Boscosa_2"]
    recursos = ["Helicoptero", "CisternaPesada"]
    riesgo_zonas = {"Zona_Critica_1": 0.95, "Zona_Boscosa_2": 0.70}
    props_recursos = {
        "Helicoptero":    {"costo_operativo": 4.0, "capacidad_mitigacion": 0.9},
        "CisternaPesada": {"costo_operativo": 2.0, "capacidad_mitigacion": 0.6},
    }
    infraestructura_bases = {
        "Base_Norte": {"tiene_helipuerto": True},
        "Base_Sur":   {"tiene_helipuerto": False},
    }
    stock_disponible = {"Helicoptero": 1, "CisternaPesada": 1}
    # [base][zona][recurso] en minutos
    matriz_transito = np.array([
        [[10.0, 30.0], [15.0, 40.0]],
        [[25.0, 20.0], [30.0, 25.0]],
    ])
    return {
        "bases": bases, "zonas": zonas, "recursos": recursos,
        "riesgo_zonas": riesgo_zonas, "props_recursos": props_recursos,
        "infraestructura_bases": infraestructura_bases,
        "stock_disponible": stock_disponible,
        "matriz_transito": matriz_transito,
    }


def extender_datos(d):
    """Agrega características de brigadas, zonas y distribución del fuego. Editar libremente."""
    # Brigadas / recursos
    d["props_recursos"]["Helicoptero"].update({
        "tiempo_preparacion": 5.0,      # min antes de despegar
        "capacidad_ha_h": 12.0,         # hectáreas que puede contener por hora
        "viento_max_kmh": 50.0,         # no opera por encima de este viento
        "requiere_acceso_terrestre": False,
        "aptitud_vegetacion": {"bosque_denso": 0.9, "pastizal": 1.0, "matorral": 0.9},
    })
    d["props_recursos"]["CisternaPesada"].update({
        "tiempo_preparacion": 10.0,
        "capacidad_ha_h": 5.0,
        "viento_max_kmh": 999.0,
        "requiere_acceso_terrestre": True,
        "aptitud_vegetacion": {"bosque_denso": 0.5, "pastizal": 1.0, "matorral": 0.8},
    })
    # Características de cada zona
    d["props_zonas"] = {
        "Zona_Critica_1": {"vegetacion": "pastizal",     "acceso_terrestre": True,  "viento_kmh": 25.0},
        "Zona_Boscosa_2": {"vegetacion": "bosque_denso", "acceso_terrestre": False, "viento_kmh": 55.0},
    }
    # Distribución del fuego por zona
    d["fuego_zonas"] = {
        "Zona_Critica_1": {"area_ha": 20.0, "intensidad": 0.8, "propagacion_ha_h": 10.0},
        "Zona_Boscosa_2": {"area_ha": 35.0, "intensidad": 0.6, "propagacion_ha_h": 6.0},
    }
    # Máximo de recursos que cada base puede despachar a la vez
    d["capacidad_base"] = {"Base_Norte": 2, "Base_Sur": 1}
    # Pesos de la función objetivo (todos los términos normalizados ~[0,1])
    d["pesos"] = {"costo": 0.25, "tiempo": 0.25, "beneficio": 1.0, "sinergia": 0.3}
    return d


# ---------- 2. MODELO DE COEFICIENTES ----------

def es_factible(d, b, z, r):
    """Poda a priori: variables imposibles no se crean (menos qubits)."""
    pr, pz = d["props_recursos"][r], d["props_zonas"][z]
    if r == "Helicoptero" and not d["infraestructura_bases"][b]["tiene_helipuerto"]:
        return False
    if pr["requiere_acceso_terrestre"] and not pz["acceso_terrestre"]:
        return False
    return pz["viento_kmh"] <= pr["viento_max_kmh"]


def coeficiente_lineal(d, b, z, r, t_max, c_max):
    """Costo + demora - beneficio esperado de enviar el recurso r desde b a z."""
    i, j, k = d["bases"].index(b), d["zonas"].index(z), d["recursos"].index(r)
    pr, pz, fz, w = d["props_recursos"][r], d["props_zonas"][z], d["fuego_zonas"][z], d["pesos"]

    t = d["matriz_transito"][i, j, k] + pr["tiempo_preparacion"]
    area_llegada = fz["area_ha"] + fz["propagacion_ha_h"] * t / 60.0   # el fuego crece mientras llega
    cobertura = min(1.0, pr["capacidad_ha_h"] / area_llegada)
    beneficio = (d["riesgo_zonas"][z] * fz["intensidad"] * pr["capacidad_mitigacion"]
                 * pr["aptitud_vegetacion"][pz["vegetacion"]] * cobertura)

    return w["costo"] * pr["costo_operativo"] / c_max + w["tiempo"] * t / t_max - w["beneficio"] * beneficio


def construir_problema(d):
    qp = QuadraticProgram(name="Brigadas_Incendio")
    combos = [(b, z, r) for b in d["bases"] for z in d["zonas"] for r in d["recursos"] if es_factible(d, b, z, r)]
    nombre = lambda b, z, r: f"x_{b}_{z}_{r}"
    for c in combos:
        qp.binary_var(name=nombre(*c))

    t_max = max(d["matriz_transito"].max() + p["tiempo_preparacion"] for p in d["props_recursos"].values())
    c_max = max(p["costo_operativo"] for p in d["props_recursos"].values())
    lineal = {nombre(*c): coeficiente_lineal(d, *c, t_max, c_max) for c in combos}

    # Término cuadrático: sinergia (aire + tierra en la misma zona) reduce el costo
    cuad = {}
    for z in d["zonas"]:
        for (b1, r1), (b2, r2) in itertools.combinations(
                [(b, r) for (b, zz, r) in combos if zz == z], 2):
            if r1 != r2:
                cuad[(nombre(b1, z, r1), nombre(b2, z, r2))] = -d["pesos"]["sinergia"]
    qp.minimize(linear=lineal, quadratic=cuad)

    def restr(vars_, sense, rhs, name):
        if len(vars_) > 1 or (vars_ and sense == ">=" and rhs >= 1):   # restricciones triviales se omiten
            qp.linear_constraint({nombre(*c): 1 for c in vars_}, sense, rhs, name)

    # Stock por tipo de recurso
    for r in d["recursos"]:
        restr([c for c in combos if c[2] == r], "<=", d["stock_disponible"][r], f"stock_{r}")
    # Cada recurso llega a cada zona desde una sola base
    for z in d["zonas"]:
        for r in d["recursos"]:
            restr([c for c in combos if c[1] == z and c[2] == r], "<=", 1, f"unica_base_{z}_{r}")
    # Capacidad de despacho de cada base
    for b in d["bases"]:
        restr([c for c in combos if c[0] == b], "<=", d["capacidad_base"][b], f"cap_{b}")
    # Toda zona con fuego debe tener al menos un recurso
    for z in d["zonas"]:
        restr([c for c in combos if c[1] == z], ">=", 1, f"cobertura_{z}")
    return qp, combos


# ---------- 3. EJECUCIÓN ----------

def mostrar(titulo, qp, res):
    print(f"\n=== {titulo} ===")
    print(f"Factible: {res.status.name} | Objetivo: {res.fval:.4f}")
    for v, x in zip(qp.variables, res.x):
        if round(x) == 1:
            print(f"  ASIGNADO: {v.name}")


def ejecutar():
    d = extender_datos(generar_datos_sistema_emergencia())
    qp, combos = construir_problema(d)
    print(f"Variables de decisión (tras poda): {len(combos)}")
    print(qp.prettyprint())

    qubo = QuadraticProgramToQubo().convert(qp)
    print(f"Qubits del QUBO (incluye holguras): {qubo.get_num_vars()}")

    # Referencia exacta clásica para validar el resultado de QAOA
    exacto = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)
    mostrar("REFERENCIA EXACTA", qp, exacto)

    qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=100), reps=1)
    res = MinimumEigenOptimizer(qaoa).solve(qp)
    mostrar("QAOA", qp, res)


if __name__ == "__main__":
    ejecutar()
