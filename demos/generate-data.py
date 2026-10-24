def generar_datos_sistema_emergencia():
    """Genera datos simulados acotados para la demo de Qiskit."""
    bases = ["Base_Norte", "Base_Sur"]
    zonas = ["Zona_Critica_1", "Zona_Boscosa_2"]
    recursos = ["Helicoptero", "CisternaPesada"]

    riesgo_zonas = {
        "Zona_Critica_1": 0.95,
        "Zona_Boscosa_2": 0.70
    }

    props_recursos = {
        "Helicoptero":    {"costo_operativo": 4.0, "capacidad_mitigacion": 0.9},
        "CisternaPesada": {"costo_operativo": 2.0, "capacidad_mitigacion": 0.6}
    }

    infraestructura_bases = {
        "Base_Norte": {"tiene_helipuerto": True},
        "Base_Sur":   {"tiene_helipuerto": False}
    }

    stock_disponible = {
        "Helicoptero":    1,
        "CisternaPesada": 1
    }

    # Matriz de tiempos simulada [base, zona, recurso]
    # 2 bases x 2 zonas x 2 recursos = 8 variables en total
    matriz_transito = np.array([
        # Base Norte -> [Zona 1 (Heli, Cisterna), Zona 2 (Heli, Cisterna)]
        [[10.0, 30.0], [15.0, 40.0]],
        # Base Sur -> [Zona 1 (Heli, Cisterna), Zona 2 (Heli, Cisterna)]
        [[25.0, 20.0], [30.0, 25.0]]
    ])

    return {
        "bases": bases, "zonas": zonas, "recursos": recursos,
        "riesgo_zonas": riesgo_zonas, "props_recursos": props_recursos,
        "infraestructura_bases": infraestructura_bases,
        "stock_disponible": stock_disponible,
        "matriz_transito": matriz_transito
    }
