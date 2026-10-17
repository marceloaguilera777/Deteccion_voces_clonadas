"""
Detector de voz real vs voz clonada por IA (Teachable Machine, proyecto de audio).

Usa el modelo exportado de Teachable Machine en formato TensorFlow Lite:
  soundclassifier_with_metadata.tflite + labels.txt  (en la misma carpeta que este archivo)

Modos:
  python detector_voz.py vivo
      Escucha el micrófono (o VB-Cable) y muestra cada segundo REAL / CLONADA.
  python detector_voz.py evaluar --reales audios/prueba/reales --clonadas audios/prueba/clones/whatsapp
      Pasa los audios de prueba en ventanas de 1 s y calcula la matriz de confusión,
      exactitud, precisión, sensibilidad, especificidad y F1 (clase positiva = clonada).

Instalar:  pip install tensorflow numpy sounddevice librosa
"""
import argparse
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
MODELO = os.path.join(AQUI, "soundclassifier_with_metadata.tflite")
ETIQUETAS = os.path.join(AQUI, "labels.txt")
SR = 44100            # Teachable Machine graba a 44,1 kHz
UMBRAL_ALERTA = 0.80  # igual que en la página web
EXTENSIONES = (".wav", ".mp3", ".ogg", ".opus", ".m4a", ".mp4", ".aac", ".flac", ".mpeg")


def cargar_interprete():
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            sys.exit("Falta TensorFlow: pip install tensorflow")
    if not os.path.exists(MODELO):
        sys.exit(f"No encuentro el modelo: {MODELO}\n"
                 "Exporta en Teachable Machine > Exportar modelo > TensorFlow Lite > Descargar.")
    interp = Interpreter(model_path=MODELO)
    interp.allocate_tensors()
    return interp


def cargar_etiquetas():
    with open(ETIQUETAS, encoding="utf-8") as f:
        # cada línea viene como "0 Background Noise", "1 Voz_Real", ...
        nombres = [l.strip().split(" ", 1)[-1] for l in f if l.strip()]
    tipos = []
    for n in nombres:
        n2 = n.lower()
        tipos.append("clon" if "clon" in n2 else "real" if "real" in n2 else "ruido")
    return nombres, tipos


class Detector:
    def __init__(self):
        self.interp = cargar_interprete()
        self.nombres, self.tipos = cargar_etiquetas()
        self.entrada = self.interp.get_input_details()[0]
        self.salida = self.interp.get_output_details()[0]
        forma = self.entrada["shape"]
        if len(forma) != 2:
            sys.exit(f"El modelo espera una entrada con forma {forma}; este script trabaja con audio crudo [1, N].")
        self.n = int(forma[1])  # muestras por ventana (~1 s)

    def predecir(self, ventana):
        x = np.asarray(ventana, dtype=np.float32).reshape(1, self.n)
        self.interp.set_tensor(self.entrada["index"], x)
        self.interp.invoke()
        return self.interp.get_tensor(self.salida["index"])[0]

    def prob_clon(self, p):
        return float(sum(v for v, t in zip(p, self.tipos) if t == "clon"))


def modo_vivo(det, dispositivo):
    import sounddevice as sd
    print("Escuchando... (Ctrl+C para salir)")
    print("Clases:", ", ".join(det.nombres))
    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32", device=dispositivo) as st:
            while True:
                bloque, _ = st.read(det.n)
                p = det.predecir(bloque[:, 0])
                mejor = int(np.argmax(p))
                pc = det.prob_clon(p)
                if pc >= UMBRAL_ALERTA:
                    veredicto = f"🚨 VOZ CLONADA ({pc:.0%})"
                elif det.tipos[mejor] == "real":
                    veredicto = f"✅ VOZ REAL ({p[mejor]:.0%})"
                elif det.tipos[mejor] == "clon":
                    veredicto = f"🤔 DUDOSA (clonada {pc:.0%}, bajo el umbral)"
                else:
                    veredicto = "… silencio / ruido"
                barras = "  ".join(f"{n}: {v:5.1%}" for n, v in zip(det.nombres, p))
                print(f"{veredicto:<40} | {barras}")
    except KeyboardInterrupt:
        print("\nListo.")


def ventanas_de(ruta, n):
    import librosa
    audio, _ = librosa.load(ruta, sr=SR, mono=True)
    for i in range(0, len(audio) - n + 1, n):  # ventanas de 1 s sin solapamiento
        yield audio[i:i + n]


def listar(carpeta):
    return sorted(os.path.join(carpeta, f) for f in os.listdir(carpeta) if f.lower().endswith(EXTENSIONES))


def modo_evaluar(det, carpeta_reales, carpeta_clonadas):
    # filas = clase real; columnas = lo que dijo el modelo
    conteo = {"real": {"real": 0, "clon": 0}, "clon": {"real": 0, "clon": 0}}
    for esperado, carpeta in (("real", carpeta_reales), ("clon", carpeta_clonadas)):
        for ruta in listar(carpeta):
            aciertos = total = 0
            for v in ventanas_de(ruta, det.n):
                pred = det.tipos[int(np.argmax(det.predecir(v)))]
                if pred == "ruido":  # silencios entre frases: no se cuentan
                    continue
                conteo[esperado][pred] += 1
                total += 1
                aciertos += pred == esperado
            print(f"{os.path.basename(ruta):<40} {aciertos}/{total} ventanas correctas")

    VP, FN = conteo["clon"]["clon"], conteo["clon"]["real"]
    FP, VN = conteo["real"]["clon"], conteo["real"]["real"]
    div = lambda a, b: a / b if b else float("nan")
    exactitud = div(VP + VN, VP + VN + FP + FN)
    precision = div(VP, VP + FP)
    sensibilidad = div(VP, VP + FN)
    especificidad = div(VN, VN + FP)
    f1 = div(2 * precision * sensibilidad, precision + sensibilidad)

    print("\nMatriz de confusión (filas = clase real, columnas = predicha)")
    print(f"{'':<14}{'real':>8}{'clonada':>10}")
    print(f"{'Voz real':<14}{VN:>8}{FP:>10}")
    print(f"{'Voz clonada':<14}{FN:>8}{VP:>10}")
    print(f"\nVP={VP} VN={VN} FP={FP} FN={FN}")
    print(f"Exactitud     = {exactitud:.1%}")
    print(f"Precisión     = {precision:.1%}")
    print(f"Sensibilidad  = {sensibilidad:.1%}")
    print(f"Especificidad = {especificidad:.1%}")
    print(f"F1            = {f1:.3f}")


def main():
    ap = argparse.ArgumentParser(description="Detector de voz real vs clonada (Teachable Machine TFLite)")
    sub = ap.add_subparsers(dest="modo", required=True)
    v = sub.add_parser("vivo", help="escuchar el micrófono en tiempo real")
    v.add_argument("--dispositivo", type=int, default=None,
                   help="número del micrófono (ver: python -m sounddevice)")
    e = sub.add_parser("evaluar", help="evaluar carpetas de audios de prueba")
    e.add_argument("--reales", required=True)
    e.add_argument("--clonadas", required=True)
    args = ap.parse_args()

    det = Detector()
    if args.modo == "vivo":
        modo_vivo(det, args.dispositivo)
    else:
        modo_evaluar(det, args.reales, args.clonadas)


if __name__ == "__main__":
    main()
