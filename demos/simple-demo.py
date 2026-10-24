# ==========================================
# MINI DEMO: MVP DE DISTRIBUCIÓN CON QISKIT
# ==========================================

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer

def ejecutar_demo_qiskit():
    print("--- 1. Inicializando el Programa Cuadrático ---")
    qp = QuadraticProgram(name="Distribucion_Heterogenea_Brigadas")

    # Definición de Variables de Decisión Binarias (x_i_k)
    # x_0_0: Asignar Helicóptero a Base 1
    # x_0_1: Asignar Cisterna a Base 1
    # x_1_0: Asignar Helicóptero a Base 2
    # x_1_1: Asignar Cisterna a Base 2
    variables = ["H_Base1", "C_Base1", "H_Base2", "C_Base2"]
    for v in variables:
        qp.binary_var(name=v)

    print(f"Variables binarias creadas: {variables}")

    print("\n--- 2. Configurando la Función Objetivo (Minimizar Costo / Riesgo) ---")
    # Coeficientes de costos ponderados por tiempo de tránsito y riesgo de zona
    # (Simulando penalizaciones de costo de despliegue y efectividad)
    lineal = {
        "H_Base1": 2.5,  # Costo operativo helicóptero base 1
        "C_Base1": 1.0,  # Costo operativo cisterna base 1
        "H_Base2": 3.0,  # Costo operativo helicóptero base 2
        "C_Base2": 1.2   # Costo operativo cisterna base 2
    }
    qp.minimize(linear=lineal)

    print("\n--- 3. Añadiendo Restricciones Duras ---")
    # Restricción 1: Solo tenemos 1 Helicóptero disponible en total para todo el territorio
    qp.linear_constraint(linear={"H_Base1": 1, "H_Base2": 1}, sense="<=", rhs=1, name="stock_helicoptero")

    # Restricción 2: Solo tenemos 1 Camión Cisterna disponible en total
    qp.linear_constraint(linear={"C_Base1": 1, "C_Base2": 1}, sense="<=", rhs=1, name="stock_cisterna")

    # Restricción 3: Base 1 solo soporta un recurso (por falta de espacio o infraestructura)
    qp.linear_constraint(linear={"H_Base1": 1, "C_Base1": 1}, sense="<=", rhs=1, name="capacidad_base1")

    # >>> NUEVA RESTRICCIÓN: Obligar a desplegar al menos un recurso en total para mitigar el riesgo <<<
    qp.linear_constraint(linear={"H_Base1": 1, "C_Base1": 1, "H_Base2": 1, "C_Base2": 1}, sense=">=", rhs=1, name="cobertura_obligatoria")

    print("Restricciones de inventario y capacidad añadidas con éxito.")

    print("\n--- 4. Convirtiendo a QUBO ---</h2>")
    # Convertidor para manejar restricciones en formato QUBO apto para algoritmos cuánticos
    qubo_converter = QuadraticProgramToQubo()
    qubo_problem = qubo_converter.convert(qp)
    print(f"Número de variables en el QUBO (incluyendo holguras): {qubo_problem.get_num_vars()}")

    print("\n--- 5. Configurando el Algoritmo Cuántico (QAOA) ---")
    # Configuramos el Sampler primitivo y el optimizador clásico COBYLA para QAOA
    sampler = Sampler()
    optimizer = COBYLA(maxiter=100)

    # Instanciamos QAOA con p=1 (número de capas del circuito)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=1)

    # Envolvemos QAOA en el optimizador de eigenvalores mínimos
    optimizer_cuantico = MinimumEigenOptimizer(qaoa)

    print("Ejecutando circuito cuántico / simulador QAOA...")
    resultado = optimizer_cuantico.solve(qubo_problem)

    print("\n==========================================")
    print("          RESULTADO DE LA DEMO            ")
    print("==========================================")
    print(f"Estado Óptimo Encontrado (Bits): {resultado.x}")
    print(f"Valor de la Función Objetivo: {resultado.fval}")

    print("\nInterpretación Operativa de la Distribución:")
    for i, var_name in enumerate(variables):
        asignado = int(resultado.x[i])
        print(f" - [{var_name}]: {'ASIGNADO (1)' if asignado == 1 else 'NO ASIGNADO (0)'}")

if __name__ == "__main__":
    ejecutar_demo_qiskit()
