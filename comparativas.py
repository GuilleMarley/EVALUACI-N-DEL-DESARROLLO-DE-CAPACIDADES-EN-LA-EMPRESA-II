import tkinter as tk
from tkinter import messagebox, Toplevel, scrolledtext
import subprocess
import whisper
import text2emotion as te
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import threading
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import traceback

# Descargar recursos NLTK
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    analyzer_vader = SentimentIntensityAnalyzer()
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    print("Error al inicializar NLTK:")
    traceback.print_exc()
    STOPWORDS = set()

# Cargar modelo RoBERTa para sentimiento y extraer etiquetas dinámicas
try:
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer_roberta = AutoTokenizer.from_pretrained(model_name)
    model_roberta = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier_roberta = pipeline('sentiment-analysis', model=model_roberta, tokenizer=tokenizer_roberta)
    labels_roberta = model_roberta.config.id2label
except Exception:
    print("Error al cargar modelo RoBERTa:")
    traceback.print_exc()
    labels_roberta = {0: 'NEG', 1: 'NEU', 2: 'POS'}

# === TRANSCRIPCIÓN AUTOMÁTICA YOUTUBE via API ===
def obtener_transcripcion_youtube_api(url):
    try:
        if 'youtu.be' in url:
            video_id = url.split('/')[-1]
        else:
            video_id = url.split('v=')[-1].split('&')[0]
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        texto = '\n'.join(seg['text'] for seg in transcript_list)
        return texto
    except Exception:
        print("Error al obtener transcripción automática de YouTube:")
        traceback.print_exc()
        raise RuntimeError('No existe transcripción automática para este video.')

# === DESCARGA Y TRANSCRIPCIÓN CON WHISPER ===
def descargar_audio_youtube(url, nombre_archivo):
    try:
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_stream.download(filename=nombre_archivo)
        return nombre_archivo
    except Exception:
        print("Error al descargar audio de YouTube:")
        traceback.print_exc()
        raise

def descargar_audio_twitch(url, nombre_archivo):
    try:
        comando = f"streamlink {url} best -o {nombre_archivo}"
        subprocess.call(comando, shell=True)
        return nombre_archivo
    except Exception:
        print("Error al descargar audio de Twitch:")
        traceback.print_exc()
        raise

def transcribir_audio_whisper(nombre_archivo):
    try:
        modelo = whisper.load_model('base')
        resultado = modelo.transcribe(nombre_archivo)
        return resultado['text']
    except Exception:
        print("Error al transcribir audio con Whisper:")
        traceback.print_exc()
        raise

# === PREPROCESAMIENTO: REMOVER STOPWORDS ===
def remover_stopwords(texto):
    try:
        tokens = word_tokenize(texto)
        filtered = [t for t in tokens if t.lower() not in STOPWORDS]
        return " ".join(filtered)
    except Exception:
        return texto

# === ANÁLISIS DE EMOCIONES con Text2Emotion ===
def analizar_emociones_text2emotion(texto):
    try:
        texto_filtro = remover_stopwords(texto)
        return te.get_emotion(texto_filtro)
    except Exception:
        print("Error al analizar emociones con Text2Emotion:")
        traceback.print_exc()
        raise

# === ANÁLISIS DE SENTIMIENTO con RoBERTa (fragmentando texto largo) ===
def analizar_sentimiento_roberta(texto, max_words_per_chunk=200):
    try:
        texto_filtro = remover_stopwords(texto)
        words = texto_filtro.split()
        fragmentos = [" ".join(words[i:i+max_words_per_chunk]) for i in range(0, len(words), max_words_per_chunk)]
        resultados = []
        for frag in fragmentos:
            res = classifier_roberta(frag)
            for r in res:
                label_id = int(r['label'].split('_')[-1])
                r['label'] = labels_roberta.get(label_id, r['label'])
                resultados.append(r)
        return resultados
    except Exception:
        print("Error al analizar sentimiento con RoBERTa:")
        traceback.print_exc()
        raise

# === ANÁLISIS DE SENTIMIENTO con VADER ===
def analizar_sentimiento_vader(texto):
    try:
        texto_filtro = remover_stopwords(texto)
        return analyzer_vader.polarity_scores(texto_filtro)
    except Exception:
        print("Error al analizar sentimiento con VADER:")
        traceback.print_exc()
        raise

# === PROCESAMIENTO GENERAL ===
def procesar_url(url, usar_youtube, data_container):
    nombre_archivo = 'audio_temp.mp4'
    try:
        # 1. Obtener transcripción (API YouTube si está seleccionado y hay subtítulos disponibles)
        if usar_youtube and ('youtube.com' in url or 'youtu.be' in url):
            texto = obtener_transcripcion_youtube_api(url)
        else:
            # Descargar audio y transcribir con Whisper
            if 'youtube.com' in url or 'youtu.be' in url:
                descargar_audio_youtube(url, nombre_archivo)
            elif 'twitch.tv' in url:
                descargar_audio_twitch(url, nombre_archivo)
            else:
                raise ValueError("URL no válida. Solo YouTube y Twitch.")
            texto = transcribir_audio_whisper(nombre_archivo)
            os.remove(nombre_archivo)

        # Guardar transcripción en el contenedor
        data_container['texto_transcripcion'] = texto

        # 2. Análisis Text2Emotion
        data_container['emociones_text2emotion'] = analizar_emociones_text2emotion(texto)

        # 3. Análisis RoBERTa (fragmentando texto largo)
        data_container['sentimiento_roberta'] = analizar_sentimiento_roberta(texto)

        # 4. Análisis VADER
        data_container['sentimiento_vader'] = analizar_sentimiento_vader(texto)

    except Exception as e:
        print("Error en procesar_url:")
        traceback.print_exc()
        raise e

# === FUNCIONES AUXILIARES para mostrar gráficos en popups ===
def crear_popup_grafico(figura):
    popup = Toplevel()
    popup.title("Gráfico")
    canvas = FigureCanvasTkAgg(figura, master=popup)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas.draw()

# === FUNCIONES DE INTERFAZ para mostrar resultados ===
def mostrar_text2emotion(data_container):
    try:
        if 'emociones_text2emotion' not in data_container:
            messagebox.showinfo("Info", "Ejecuta primero el análisis.")
            return
        emociones = data_container['emociones_text2emotion']
        fig, ax = plt.subplots(figsize=(4,2))
        ax.bar(list(emociones.keys()), list(emociones.values()), color='skyblue')
        ax.set_title("Text2Emotion")
        ax.set_ylabel("Puntuación")
        crear_popup_grafico(fig)
    except Exception:
        print("Error en mostrar_text2emotion:")
        traceback.print_exc()
        messagebox.showerror("Error", "Falló al mostrar resultados Text2Emotion.")

def mostrar_roberta(data_container):
    try:
        if 'sentimiento_roberta' not in data_container:
            messagebox.showinfo("Info", "Ejecuta primero el análisis.")
            return
        resultados = data_container['sentimiento_roberta']
        agr = {}
        for r in resultados:
            label = r['label']
            agr[label] = agr.get(label, 0) + r['score']
        fig, ax = plt.subplots(figsize=(4,2))
        ax.bar(list(agr.keys()), list(agr.values()), color='lightgreen')
        ax.set_title("RoBERTa")
        ax.set_ylabel("Score acumulado")
        crear_popup_grafico(fig)
    except Exception:
        print("Error en mostrar_roberta:")
        traceback.print_exc()
        messagebox.showerror("Error", "Falló al mostrar resultados RoBERTa.")

def mostrar_vader(data_container):
    try:
        if 'sentimiento_vader' not in data_container:
            messagebox.showinfo("Info", "Ejecuta primero el análisis.")
            return
        vader_score = data_container['sentimiento_vader']
        fig, ax = plt.subplots(figsize=(4,2))
        ax.bar(list(vader_score.keys()), list(vader_score.values()), color='salmon')
        ax.set_title("VADER")
        ax.set_ylabel("Score")
        crear_popup_grafico(fig)
    except Exception:
        print("Error en mostrar_vader:")
        traceback.print_exc()
        messagebox.showerror("Error", "Falló al mostrar resultados VADER.")

def mostrar_comparativa(data_container):
    try:
        if not all(k in data_container for k in ['emociones_text2emotion','sentimiento_roberta','sentimiento_vader']):
            messagebox.showinfo("Info", "Ejecuta primero el análisis completo.")
            return
        emo = data_container['emociones_text2emotion']
        rob = data_container['sentimiento_roberta']
        vader = data_container['sentimiento_vader']

        # Agrupar RoBERTa
        agr_rob = {}
        for r in rob:
            label = r['label']
            agr_rob[label] = agr_rob.get(label, 0) + r['score']

        # Convertir Text2Emotion a Pos/Neu/Neg
        pos_t2e = emo.get('Happy',0) + emo.get('Surprise',0)
        neg_t2e = emo.get('Sad',0) + emo.get('Angry',0) + emo.get('Fear',0)
        neu_t2e = max(0, 1 - pos_t2e - neg_t2e)

        fig, ax = plt.subplots(figsize=(8,4))
        labels = ['Positivo','Neutro','Negativo']
        t2e_vals = [pos_t2e, neu_t2e, neg_t2e]
        rob_vals = [agr_rob.get('POS',0), agr_rob.get('NEU',0), agr_rob.get('NEG',0)]
        vader_vals = [vader.get('pos',0), vader.get('neu',0), vader.get('neg',0)]

        x = range(len(labels))
        ax.bar([p - 0.2 for p in x], t2e_vals, width=0.2, label='Text2Emotion', color='skyblue')
        ax.bar(x, rob_vals, width=0.2, label='RoBERTa', color='lightgreen')
        ax.bar([p + 0.2 for p in x], vader_vals, width=0.2, label='VADER', color='salmon')

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Score/ Proporción')
        ax.set_title('Comparativa de Métodos de Análisis')
        ax.legend()
        crear_popup_grafico(fig)
    except Exception:
        print("Error en mostrar_comparativa:")
        traceback.print_exc()
        messagebox.showerror("Error", "Falló al mostrar gráfica comparativa.")

# === FUNCIONES ASÍNCRONAS E INTERFAZ ===
def iniciar_proceso(entry_url, var_yt, lbl_status, data_container, cuadro_transcripcion):
    url = entry_url.get().strip()
    if not url:
        messagebox.showwarning("Atención", "Introduce una URL.")
        return
    lbl_status.config(text="Procesando, por favor espere...")
    threading.Thread(
        target=lambda: run_background(url, var_yt.get(), lbl_status, data_container, cuadro_transcripcion),
        daemon=True
    ).start()

def run_background(url, usar_youtube, lbl_status, data_container, cuadro_transcripcion):
    try:
        procesar_url(url, usar_youtube, data_container)
        # Mostrar la transcripción en el cuadro principal
        texto = data_container.get('texto_transcripcion', '')
        cuadro_transcripcion.delete('1.0', tk.END)
        cuadro_transcripcion.insert(tk.END, texto)
        lbl_status.config(text="Procesamiento completado.")
    except Exception as e:
        lbl_status.config(text="Error en procesamiento.")
        print("Error en run_background:")
        traceback.print_exc()
        messagebox.showerror("Error", str(e))

def interfaz():
    try:
        ventana = tk.Tk()
        ventana.title('Comparativo Sentimiento: Text2Emotion / RoBERTa / VADER')
        ventana.geometry('900x850')
        ventana.configure(bg='#f0f0f0')

        # Asegurar que cerrar la ventana termine el script
        ventana.protocol("WM_DELETE_WINDOW", lambda: (ventana.destroy(), sys.exit()))

        tk.Label(ventana, text='URL YouTube o Twitch:', bg='#f0f0f0', font=('Arial',12)).pack(pady=5)
        entry_url = tk.Entry(ventana, width=90, font=('Arial',11))
        entry_url.pack(pady=3)

        var_yt_check = tk.IntVar()
        tk.Checkbutton(
            ventana,
            text='Usar subtítulos YouTube (si existen)',
            variable=var_yt_check,
            bg='#f0f0f0',
            font=('Arial',10)
        ).pack()

        lbl_status = tk.Label(ventana, text='Esperando URL...', bg='#f0f0f0', fg='blue', font=('Arial',10))
        lbl_status.pack(pady=5)

        tk.Button(
            ventana,
            text='Iniciar Análisis',
            bg='#2196F3',
            fg='white',
            font=('Arial',12),
            command=lambda: iniciar_proceso(entry_url, var_yt_check, lbl_status, data_container, cuadro_transcripcion)
        ).pack(pady=8)

        tk.Label(ventana, text='Transcripción Completa:', bg='#f0f0f0', font=('Arial',12,'underline')).pack(pady=5)

        global cuadro_transcripcion
        cuadro_transcripcion = scrolledtext.ScrolledText(ventana, height=10, width=100, font=('Courier',10))
        cuadro_transcripcion.pack(pady=3)

        tk.Label(ventana, text='Resultados por Gráficos:', bg='#f0f0f0', font=('Arial',12,'underline')).pack(pady=5)

        frame_botones = tk.Frame(ventana, bg='#f0f0f0')
        frame_botones.pack(pady=5)

        tk.Button(
            frame_botones,
            text='Text2Emotion',
            bg='#03A9F4',
            fg='white',
            font=('Arial',10),
            command=lambda: mostrar_text2emotion(data_container)
        ).grid(row=0, column=0, padx=5)

        tk.Button(
            frame_botones,
            text='RoBERTa',
            bg='#8BC34A',
            fg='white',
            font=('Arial',10),
            command=lambda: mostrar_roberta(data_container)
        ).grid(row=0, column=1, padx=5)

        tk.Button(
            frame_botones,
            text='VADER',
            bg='#FF5722',
            fg='white',
            font=('Arial',10),
            command=lambda: mostrar_vader(data_container)
        ).grid(row=0, column=2, padx=5)

        tk.Button(
            frame_botones,
            text='Comparar',
            bg='#9C27B0',
            fg='white',
            font=('Arial',10),
            command=lambda: mostrar_comparativa(data_container)
        ).grid(row=0, column=3, padx=5)

        ventana.mainloop()
    except Exception:
        print("Error en interfaz principal:")
        traceback.print_exc()

# Diccionario global para almacenar resultados
data_container = {}

if __name__ == '__main__':
    interfaz()
