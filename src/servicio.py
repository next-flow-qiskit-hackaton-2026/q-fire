# ==========================================================
# SERVICIO: ASIGNACIÓN DE BRIGADAS A ZONAS DE INCENDIO (QAOA)
# ==========================================================
# Colab:  !pip install qiskit qiskit-optimization qiskit-algorithms
#
# Uso: llamar a asignar_brigadas(...) con los datos de entrada. Devuelve un dict con el resultado,
# así se pueden cambiar valores y comparar salidas. La celda de ejemplo para Colab está al final.

import itertools
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA

PESOS_DEFECTO = {"costo": 0.25, "tiempo": 0.25, "beneficio": 1.0, "sinergia": 0.3}
CLAVES_RECURSO = ["costo_operativo", "capacidad_mitigacion", "tiempo_preparacion", "capacidad_ha_h",
                  "viento_max_kmh", "requiere_acceso_terrestre", "aptitud_vegetacion"]
CLAVES_ZONA = ["vegetacion", "acceso_terrestre", "viento_kmh"]
CLAVES_FUEGO = ["area_ha", "intensidad", "propagacion_ha_h"]


def _validar(d):
    """Valida la entrada: es el borde del servicio, los errores deben ser claros."""
    n_b, n_z, n_r = len(d["bases"]), len(d["zonas"]), len(d["recursos"])
    if d["matriz_transito"].shape != (n_b, n_z, n_r):
        raise ValueError(f"matriz_transito debe tener forma {(n_b, n_z, n_r)} [base, zona, recurso], "
                         f"recibida {d['matriz_transito'].shape}")
    for r in d["recursos"]:
        faltan = [k for k in CLAVES_RECURSO if k not in d["props_recursos"][r]]
        if faltan:
            raise ValueError(f"props_recursos['{r}'] sin: {faltan}")
        if r not in d["stock_disponible"]:
            raise ValueError(f"stock_disponible sin '{r}'")
    for z in d["zonas"]:
        for nombre, claves in (("props_zonas", CLAVES_ZONA), ("fuego_zonas", CLAVES_FUEGO)):
            faltan = [k for k in claves if k not in d[nombre][z]]
            if faltan:
                raise ValueError(f"{nombre}['{z}'] sin: {faltan}")
        if z not in d["riesgo_zonas"]:
            raise ValueError(f"riesgo_zonas sin '{z}'")
        veg = d["props_zonas"][z]["vegetacion"]
        for r in d["recursos"]:
            if veg not in d["props_recursos"][r]["aptitud_vegetacion"]:
                raise ValueError(f"aptitud_vegetacion de '{r}' sin la vegetación '{veg}'")
    for b in d["bases"]:
        if b not in d["infraestructura_bases"] or b not in d["capacidad_base"]:
            raise ValueError(f"infraestructura_bases/capacidad_base sin '{b}'")


def _es_factible(d, b, z, r):
    pr, pz = d["props_recursos"][r], d["props_zonas"][z]
    if d["disponibilidad"] and d["disponibilidad"].get(b, {}).get(r, 0) <= 0:   # la base no tiene ese recurso
        return False
    if r == "Helicoptero" and not d["infraestructura_bases"][b]["tiene_helipuerto"]:
        return False
    if pr["requiere_acceso_terrestre"] and not pz["acceso_terrestre"]:
        return False
    return pz["viento_kmh"] <= pr["viento_max_kmh"]


def _coeficiente(d, b, z, r, t_max, c_max):
    i, j, k = d["bases"].index(b), d["zonas"].index(z), d["recursos"].index(r)
    pr, pz, fz, w = d["props_recursos"][r], d["props_zonas"][z], d["fuego_zonas"][z], d["pesos"]
    t = d["matriz_transito"][i, j, k] + pr["tiempo_preparacion"]
    area_llegada = fz["area_ha"] + fz["propagacion_ha_h"] * t / 60.0   # el fuego crece mientras llega
    cobertura = min(1.0, pr["capacidad_ha_h"] / area_llegada)
    beneficio = (d["riesgo_zonas"][z] * fz["intensidad"] * pr["capacidad_mitigacion"]
                 * pr["aptitud_vegetacion"][pz["vegetacion"]] * cobertura)
    return w["costo"] * pr["costo_operativo"] / c_max + w["tiempo"] * t / t_max - w["beneficio"] * beneficio


def _construir(d):
    qp = QuadraticProgram(name="Brigadas_Incendio")
    combos = [(b, z, r) for b in d["bases"] for z in d["zonas"] for r in d["recursos"] if _es_factible(d, b, z, r)]
    nombre = lambda b, z, r: f"x_{b}_{z}_{r}"
    for c in combos:
        qp.binary_var(name=nombre(*c))
    if not combos:
        return qp, combos

    t_max = max(d["matriz_transito"].max() + p["tiempo_preparacion"] for p in d["props_recursos"].values())
    c_max = max(p["costo_operativo"] for p in d["props_recursos"].values())
    lineal = {nombre(*c): _coeficiente(d, *c, t_max, c_max) for c in combos}

    cuad = {}   # sinergia: recurso distinto en la misma zona
    for z in d["zonas"]:
        for (b1, r1), (b2, r2) in itertools.combinations([(b, r) for (b, zz, r) in combos if zz == z], 2):
            if r1 != r2:
                cuad[(nombre(b1, z, r1), nombre(b2, z, r2))] = -d["pesos"]["sinergia"]
    qp.minimize(linear=lineal, quadratic=cuad)

    def restr(vars_, sense, rhs, name):
        if len(vars_) > 1 or (vars_ and sense == ">=" and rhs >= 1):   # triviales se omiten
            qp.linear_constraint({nombre(*c): 1 for c in vars_}, sense, rhs, name)

    if d["disponibilidad"]:
        # cada brigada (base, tipo) va a una sola zona; reemplaza a stock y capacidad_base (que quedarían redundantes)
        for b in d["bases"]:
            for r in d["recursos"]:
                restr([c for c in combos if c[0] == b and c[2] == r], "<=",
                      d["disponibilidad"].get(b, {}).get(r, 0), f"disp_{b}_{r}")
    else:
        for r in d["recursos"]:
            restr([c for c in combos if c[2] == r], "<=", d["stock_disponible"][r], f"stock_{r}")
        for b in d["bases"]:
            restr([c for c in combos if c[0] == b], "<=", d["capacidad_base"][b], f"cap_{b}")
    for z in d["zonas"]:
        for r in d["recursos"]:
            restr([c for c in combos if c[1] == z and c[2] == r], "<=", 1, f"unica_base_{z}_{r}")
    if d["max_una_por_zona"]:
        for z in d["zonas"]:
            restr([c for c in combos if c[1] == z], "<=", 1, f"max1_{z}")
    if d["cobertura_obligatoria"]:
        for z in d["zonas"]:
            restr([c for c in combos if c[1] == z], ">=", 1, f"cobertura_{z}")
    return qp, combos


def _resumen(combos, res):
    # res.x sigue el orden de creación de las variables, que es el de combos
    ok = res.status.name == "SUCCESS"   # una solución infactible no se reporta como asignación
    return {"estado": res.status.name, "objetivo": round(float(res.fval), 6) if ok else None,
            "asignaciones": [{"base": b, "zona": z, "recurso": r}
                             for (b, z, r), x in zip(combos, res.x) if ok and round(x) == 1]}


def asignar_brigadas(bases, zonas, recursos, riesgo_zonas, props_recursos, infraestructura_bases,
                     stock_disponible, matriz_transito, props_zonas, fuego_zonas, capacidad_base,
                     pesos=None, usar_qaoa=True, reps=1, maxiter=100, max_qubits=20,
                     disponibilidad=None, max_una_por_zona=False, cobertura_obligatoria=True):
    """Resuelve la asignación y devuelve un dict serializable a JSON (json.dumps / respuesta de API).

    Los ids de base, zona y recurso en la respuesta son exactamente los strings recibidos en
    `bases`, `zonas` y `recursos`, así que pueden ser los ids reales del sistema.

    Respuesta:
      estado:               "OK" | "SIN_VARIABLES" (la poda eliminó todas las opciones)
      metodo:               "qaoa" | "exacto" (quién produjo `asignaciones`)
      asignaciones:         [{"base": id, "zona": id, "recurso": id}, ...]
      objetivo:             valor a minimizar (más bajo = mejor; unidades normalizadas)
      zonas_sin_cobertura:  zonas donde ningún recurso puede operar
      opciones_factibles:   combinaciones base/zona/recurso que superaron la poda
      qubits:               tamaño del QUBO
      qaoa_coincide_con_exacto: bool | None
      exacto / qaoa:        detalle de cada solver (estado, objetivo, asignaciones) | None

    matriz_transito: array [base][zona][recurso] en minutos (se acepta lista anidada).
    pesos: dict parcial; lo no indicado usa PESOS_DEFECTO.
    max_qubits: si el QUBO lo supera, se omite QAOA y `asignaciones` viene del solver exacto.
    disponibilidad: {base: {recurso: n_brigadas}}. Si se da, limita cada (base, recurso) a n envíos y
                    reemplaza a stock_disponible/capacidad_base en las restricciones.
    max_una_por_zona: como máximo 1 recurso en total por zona.
    cobertura_obligatoria: toda zona alcanzable debe recibir al menos 1 recurso (si no hay
                    suficientes brigadas, el problema es INFEASIBLE; ponerlo en False para priorizar).
    Lanza ValueError si la entrada es inválida (en una API: responder 422).
    """
    d = {"bases": bases, "zonas": zonas, "recursos": recursos, "riesgo_zonas": riesgo_zonas,
         "props_recursos": props_recursos, "infraestructura_bases": infraestructura_bases,
         "stock_disponible": stock_disponible, "matriz_transito": np.asarray(matriz_transito, dtype=float),
         "props_zonas": props_zonas, "fuego_zonas": fuego_zonas, "capacidad_base": capacidad_base,
         "pesos": {**PESOS_DEFECTO, **(pesos or {})}, "disponibilidad": disponibilidad,
         "max_una_por_zona": max_una_por_zona, "cobertura_obligatoria": cobertura_obligatoria}
    _validar(d)

    qp, combos = _construir(d)
    # Zonas donde ningún recurso puede operar: el modelo no las puede cubrir
    sin_cobertura = [z for z in zonas if not any(c[1] == z for c in combos)]
    salida = {"estado": "SIN_VARIABLES", "metodo": None, "asignaciones": [], "objetivo": None,
              "zonas_sin_cobertura": sin_cobertura, "qubits": None,
              "opciones_factibles": [{"base": b, "zona": z, "recurso": r} for b, z, r in combos],
              "qaoa_coincide_con_exacto": None, "exacto": None, "qaoa": None}
    if not combos:
        return salida

    salida["qubits"] = QuadraticProgramToQubo().convert(qp).get_num_vars()
    salida["exacto"] = _resumen(combos, MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp))
    if usar_qaoa and salida["qubits"] <= max_qubits:
        qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(maxiter=maxiter), reps=reps)
        salida["qaoa"] = _resumen(combos, MinimumEigenOptimizer(qaoa).solve(qp))
        key = lambda r: sorted(map(tuple, (a.values() for a in r["asignaciones"])))
        salida["qaoa_coincide_con_exacto"] = key(salida["qaoa"]) == key(salida["exacto"])

    # QAOA es el resultado principal salvo que no haya hallado solución factible
    usa_qaoa = bool(salida["qaoa"]) and salida["qaoa"]["estado"] == "SUCCESS"
    principal = salida["qaoa"] if usa_qaoa else salida["exacto"]
    salida.update(estado="OK" if principal["estado"] == "SUCCESS" else principal["estado"],
                  metodo="qaoa" if usa_qaoa else "exacto",
                  asignaciones=principal["asignaciones"], objetivo=principal["objetivo"])
    return salida


# ==========================================================
# CELDA PARA COLAB: copiar, rellenar/modificar valores y ejecutar
# ==========================================================
# resultado = asignar_brigadas(
#     bases=["Base_Norte", "Base_Sur"],
#     zonas=["Zona_Critica_1", "Zona_Boscosa_2"],
#     recursos=["Helicoptero", "CisternaPesada"],   # "Helicoptero" requiere helipuerto en la base
#
#     riesgo_zonas={"Zona_Critica_1": 0.95, "Zona_Boscosa_2": 0.70},
#
#     props_recursos={
#         "Helicoptero": {
#             "costo_operativo": 4.0, "capacidad_mitigacion": 0.9,
#             "tiempo_preparacion": 5.0,          # min
#             "capacidad_ha_h": 12.0,             # ha contenidas por hora
#             "viento_max_kmh": 50.0,
#             "requiere_acceso_terrestre": False,
#             "aptitud_vegetacion": {"bosque_denso": 0.9, "pastizal": 1.0, "matorral": 0.9},
#         },
#         "CisternaPesada": {
#             "costo_operativo": 2.0, "capacidad_mitigacion": 0.6,
#             "tiempo_preparacion": 10.0,
#             "capacidad_ha_h": 5.0,
#             "viento_max_kmh": 999.0,
#             "requiere_acceso_terrestre": True,
#             "aptitud_vegetacion": {"bosque_denso": 0.5, "pastizal": 1.0, "matorral": 0.8},
#         },
#     },
#
#     infraestructura_bases={"Base_Norte": {"tiene_helipuerto": True},
#                            "Base_Sur":   {"tiene_helipuerto": False}},
#     capacidad_base={"Base_Norte": 2, "Base_Sur": 1},        # máx. recursos despachados por base
#     stock_disponible={"Helicoptero": 1, "CisternaPesada": 1},
#
#     # minutos de tránsito [base][zona][recurso]
#     matriz_transito=[[[10, 30], [15, 40]],     # Base_Norte -> Zona1(Heli, Cist), Zona2(Heli, Cist)
#                      [[25, 20], [30, 25]]],    # Base_Sur   -> ...
#
#     props_zonas={
#         "Zona_Critica_1": {"vegetacion": "pastizal",     "acceso_terrestre": True,  "viento_kmh": 25.0},
#         "Zona_Boscosa_2": {"vegetacion": "bosque_denso", "acceso_terrestre": False, "viento_kmh": 55.0},
#     },
#     fuego_zonas={
#         "Zona_Critica_1": {"area_ha": 20.0, "intensidad": 0.8, "propagacion_ha_h": 10.0},
#         "Zona_Boscosa_2": {"area_ha": 35.0, "intensidad": 0.6, "propagacion_ha_h": 6.0},
#     },
#
#     # Opcionales
#     pesos={"costo": 0.25, "tiempo": 0.25, "beneficio": 1.0, "sinergia": 0.3},
#     usar_qaoa=True, reps=1, maxiter=100, max_qubits=20,
# )
# import json
# print(json.dumps(resultado, indent=2, ensure_ascii=False))
