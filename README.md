# ¿Voz real o clonada por IA? 🎙️🤖

Detector que escucha una voz y dice si es de una **persona real** o si fue **clonada con inteligencia artificial**.
El modelo se entrenó con **Teachable Machine** (proyecto de audio) y corre directamente en el navegador con **TensorFlow.js**.

**Autor:** Luis Marcelo Aguilera Salazar · Ingeniería en Inteligencia Artificial (4.º semestre) · Universidad Católica Boliviana San Pablo, Santa Cruz

---

## Contenido del repositorio

| Archivo | Qué es |
|---|---|
| `detector_voz.html` | La aplicación: carga el modelo, escucha el micrófono y muestra REAL / CLONADA. Tiene un modo de evaluación que arma la matriz de confusión y calcula las métricas. |
| `modelo/` | El modelo exportado de Teachable Machine (TensorFlow.js): `model.json`, `metadata.json` y `weights.bin`. |
| `docs/` | El artículo científico en PDF. |
| `frases_grabacion.txt` | Las 25 frases usadas: 1 a 20 para el entrenamiento y 21 a 25 para la prueba externa. |

## Cómo hacerlo funcionar

> ⚠️ El navegador no deja usar el micrófono ni cargar el modelo si abres el `.html` con doble clic (`file://`). Hay que abrirlo desde un servidor, con cualquiera de estas dos opciones.

### Opción A: en línea (sin instalar nada)

Abrir: **`[pega aquí el link de GitHub Pages]`**

1. Dejar el campo del modelo como está (`./modelo/`) y dar clic en **Cargar y escuchar**.
2. Permitir el micrófono cuando Chrome lo pida.
3. Hablar o reproducir un audio cerca del micrófono. La página muestra ✅ **VOZ REAL**, 🚨 **VOZ CLONADA** o 🤔 **DUDOSA**, con la probabilidad de cada clase.

### Opción B: en la computadora

Requisito: Python 3 (solo se usa para levantar un servidor local).

```bash
git clone [pega aquí el link del repositorio]
cd [nombre-del-repositorio]
python -m http.server 8000
```

Luego abrir en **Google Chrome**: <http://localhost:8000/detector_voz.html> y seguir los mismos pasos de la Opción A.

> Si la carpeta `modelo/` no carga, se puede pegar en el campo del modelo el link de Teachable Machine:
> `[pega aquí tu link https://teachablemachine.withgoogle.com/models/XXXX/]`

## Modo evaluación (cómo se obtuvieron los resultados)

1. En **Modo**, elegir *Evaluación (ventanas de 1 s sin solapar)*.
2. En **Clase que estás reproduciendo ahora**, elegir *Voz real* o *Voz clonada (XTTS-v2)*.
3. Reproducir los audios de prueba. Para no captar ruido del ambiente se usó el cable virtual [VB-Cable](https://vb-audio.com/Cable/): salida de Windows en *CABLE Input* y micrófono de Chrome en *CABLE Output*.
4. La página llena la matriz de confusión y calcula exactitud, precisión, sensibilidad y F1 (clase positiva = voz clonada). Los silencios entre frases no se cuentan.

## Cómo se construyó

1. **Voces reales:** 6 personas leyeron las frases 1 a 20 con su celular y las enviaron por WhatsApp (~90 s cada una).
2. **Voces clonadas:** se clonaron las mismas voces y frases con **XTTS-v2** (Hugging Face), con 8 a 10 s de referencia, y se reenviaron por WhatsApp para que tuvieran la misma compresión.
3. **Entrenamiento:** Teachable Machine (audio), con 3 clases: *Ruido de fondo* (100 muestras), *Voz real* (535) y *Voz clonada* (397), y 50 épocas.
4. **Prueba externa:** 3 personas que el modelo nunca escuchó, con las frases 21 a 25 y sus clones.

## Resultados (prueba externa, 276 ventanas de 1 s)

| Métrica | Valor |
|---|---|
| Exactitud | 64,1 % |
| Precisión | 55,2 % |
| Sensibilidad (clones detectados) | 52,7 % |
| Especificidad (reales reconocidas) | 71,7 % |
| F1 | 0,540 |

El modelo funciona por encima del azar (exactitud balanceada de 62,2 % frente a 50 %), pero con un margen moderado. Limitaciones: solo 6 voces de entrenamiento, un solo clonador (XTTS-v2), un idioma y la compresión de WhatsApp.

## Tecnologías

Teachable Machine · TensorFlow.js · `@tensorflow-models/speech-commands` · XTTS-v2 · VB-Cable

## Ética

Todas las personas participantes dieron su consentimiento para la grabación y la clonación de su voz con fines académicos. **Por eso los audios no se publican en este repositorio.**
