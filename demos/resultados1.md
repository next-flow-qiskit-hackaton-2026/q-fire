# Resultados 1 – Asignación de brigadas a zonas de incendio (QAOA)

Script: `demos/advanced-demo.py`

## 1. Resultado

| Método | Recursos asignados | Objetivo |
|---|---|---|
| Referencia exacta (clásica) | Helicóptero (Base_Norte → Zona_Critica_1) + Cisterna (Base_Sur → Zona_Critica_1) | **−0.1560** |
| QAOA | Igual | **−0.1560** |

QAOA coincide con la solución exacta. El problema tiene 3 variables de decisión y 7 qubits en el QUBO: las 3 variables más 4 qubits de holgura, que se agregan para convertir las desigualdades en igualdades y no son decisiones.

## 2. Contexto de las variables utilizadas

### Variables de decisión
`x_base_zona_recurso` es binaria: vale 1 si ese recurso sale de esa base hacia esa zona.

Hay 8 combinaciones posibles (2 bases × 2 zonas × 2 recursos). Solo 3 sobreviven a la poda en `es_factible()`:

| Variable | Tránsito (min) | Preparación (min) | Tiempo total t (min) |
|---|---|---|---|
| `x_Base_Norte_Zona_Critica_1_Helicoptero` | 10 | 5 | 15 |
| `x_Base_Norte_Zona_Critica_1_CisternaPesada` | 30 | 10 | 40 |
| `x_Base_Sur_Zona_Critica_1_CisternaPesada` | 20 | 10 | 30 |

Las 5 combinaciones restantes se descartaron:
- **Helicóptero desde Base_Sur:** la base no tiene helipuerto.
- **Toda la Zona_Boscosa_2:** el viento es de 55 km/h y el helicóptero solo opera hasta 50 km/h. La zona tampoco tiene acceso terrestre, así que la cisterna no puede llegar.

### Datos de los recursos

| | Helicóptero | Cisterna pesada |
|---|---|---|
| Costo operativo | 4.0 | 2.0 |
| Capacidad de mitigación | 0.9 | 0.6 |
| Capacidad de contención (ha/h) | 12 | 5 |
| Preparación (min) | 5 | 10 |
| Viento máximo (km/h) | 50 | sin límite |
| Aptitud en pastizal | 1.0 | 1.0 |
| Stock | 1 | 1 |

### Datos de la zona atendida (Zona_Critica_1)
- Riesgo: 0.95.
- Vegetación: pastizal.
- Acceso terrestre: sí.
- Viento: 25 km/h.
- Fuego: 20 ha, intensidad 0.8, propagación 10 ha/h.

### Otros parámetros
- Capacidad de despacho: Base_Norte 2 recursos, Base_Sur 1.
- Pesos de la función objetivo: costo 0.25, tiempo 0.25, beneficio 1.0, sinergia 0.3.
- Factores de normalización: tiempo máximo 50 min, costo máximo 4.0.

## 3. Cómo se calcula cada coeficiente

```
coef = 0.25·(costo/4) + 0.25·(t/50) − beneficio
area_llegada = 20 + 10·t/60            # el fuego crece mientras llega la brigada
cobertura    = min(1, ha_h / area_llegada)
beneficio    = riesgo · intensidad · mitigación · aptitud · cobertura
```

| Variable | Costo | Demora | Beneficio | **Coeficiente** |
|---|---|---|---|---|
| Helicóptero Norte → Z1 | 0.250 | 0.075 | 0.365 | **−0.0398** |
| Cisterna Norte → Z1 | 0.125 | 0.200 | 0.086 | **+0.2395** |
| Cisterna Sur → Z1 | 0.125 | 0.150 | 0.091 | **+0.1838** |

Término cuadrático: **−0.3** por cada par helicóptero + cisterna en la misma zona. Representa la sinergia de combinar aire y tierra.

## 4. Por qué el objetivo es negativo

El problema es de **minimización**: el solver busca el valor **más bajo** (el más negativo), no el mayor.

La función objetivo es:

```
objetivo = costos + demoras − beneficios − sinergia
```

- Los costos y las demoras **suman**, porque son malos.
- Los beneficios y la sinergia **restan**, porque son buenos.

Cuando el beneficio y la sinergia superan al costo y la demora, el total queda por debajo de cero. Por eso, **más negativo significa mejor asignación**. El signo no indica un error ni una pérdida.

Comparación de las combinaciones posibles:

| Combinación | Cálculo | Objetivo |
|---|---|---|
| Helicóptero Norte + Cisterna Sur | −0.0398 + 0.1838 − 0.3 | **−0.1560** (elegida, la más baja) |
| Helicóptero Norte + Cisterna Norte | −0.0398 + 0.2395 − 0.3 | −0.1003 |
| Solo helicóptero | −0.0398 | −0.0398 |

La primera es la mejor porque la cisterna del Sur llega antes (30 min contra 40), así que cuesta menos en demora y el fuego crece menos antes de que llegue.

Si se prefiere un objetivo a maximizar (positivo = bueno), basta con invertir el signo de todos los coeficientes. La solución óptima sería la misma.

## 5. Limitaciones a tener en cuenta

- **Zona_Boscosa_2 quedó sin atender.** La poda eliminó todas sus variables y la restricción de cobertura se omitió al no haber variables, así que el modelo no advierte que esa zona no tiene recursos. Conviene agregar un aviso en el script, o subir el `viento_max_kmh` del helicóptero para evaluar esa zona.
- Los valores del objetivo no tienen unidades reales: son costos normalizados y sirven para comparar alternativas entre sí.
- Los `SparseEfficiencyWarning` de scipy son avisos de rendimiento y no afectan el resultado.
