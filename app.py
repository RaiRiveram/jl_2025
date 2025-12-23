import streamlit as st
import pandas as pd
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pathlib import Path
# ----------------- CONFIGURACIÓN DE GRUPOS -----------------

EQUIPO_IPAE_AUTORES = [
    "E- Estefani Paima",
    "Ipae Alejandra Cam",
    "Pao Sánchez Ipae",
    "Ipae Yerson"
]

# ---------------- CONFIG STREAMLIT ----------------
st.set_page_config(page_title="Análisis del grupo de WhatsApp - Jóvenes Líderes IPAE",
                   layout="wide")
# ---------------- SEGURIDAD SIMPLE ----------------

# ---------------- SEGURIDAD SIMPLE ----------------
# ----------------- LOGIN -----------------
st.markdown("""
<style>
.login-container {
    max-width: 420px;
    margin: 120px auto;
    padding: 2rem;
    background-color: #ffffff;
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}
.login-container h2 {
    text-align: center;
    color: #111111;
}
.login-container p {
    text-align: center;
    color: #444444;
}
</style>
""", unsafe_allow_html=True)

PASSWORD = "jl2025"

st.markdown("""
<div class="login-container">
    <h2>🔐 Acceso privado</h2>
    <p>Ingresa la clave para acceder al dashboard</p>
</div>
""", unsafe_allow_html=True)
pwd = st.text_input(
    "Clave de acceso",
    type="password",
    label_visibility="collapsed"
)

btn = st.button("🔓 Ingresar", use_container_width=True)

if btn:
    if pwd == PASSWORD:
        st.session_state.auth = True
        st.experimental_rerun()
    else:
        st.error("❌ Clave incorrecta")
        st.stop()
else:
    st.stop()

#######################
st.title("📊 Análisis del grupo de WhatsApp - Jóvenes Líderes IPAE")
st.markdown("""
<style>
/* agranda texto en dataframes */
[data-testid="stDataFrame"] div {
  font-size: 15px;
}

/* centra encabezados */
[data-testid="stDataFrame"] thead tr th {
  text-align: center !important;
}

/* centra celdas */
[data-testid="stDataFrame"] tbody tr td {
  text-align: center !important;
}
</style>
""", unsafe_allow_html=True)

# Ruta al archivo dentro del proyecto
DATA_PATH = Path("data/chat_jlipae.txt")

# ----------------- PARSER ROBUSTO PARA TU FORMATO -----------------
# Ejemplo real:
# [20/08/25, 7:00:02 a. m.] Nombre: Mensaje

def parse_whatsapp(texto: str) -> pd.DataFrame:
    filas = []
    current = None

    for raw_line in texto.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Buscar el primer "[" (a veces hay caracteres invisibles antes)
        idx = line.find('[')
        if idx == -1:
            # Probable continuación de mensaje multilínea
            if current is not None:
                current["mensaje"] += "\n" + line
            continue

        if idx != 0:
            line = line[idx:]

        if not line.startswith('['):
            # Continuación
            if current is not None:
                current["mensaje"] += "\n" + line
            continue

        # Separar encabezado [fecha, hora am/pm] de "autor: mensaje"
        if '] ' not in line:
            # no está completo, lo tratamos como continuación
            if current is not None:
                current["mensaje"] += "\n" + line
            continue

        header_body = line[1:]  # quitamos "["
        header, rest = header_body.split('] ', 1)

        if ': ' not in rest:
            # Mensaje del sistema sin "autor: mensaje" -> puedes contarlo o ignorarlo
            continue

        try:
            fecha_str, hora_ampm = header.split(',', 1)
        except ValueError:
            # Algo raro en el header
            continue

        fecha_str = fecha_str.strip()
        hora_ampm = hora_ampm.strip()  # ejemplo: "7:00:02 a. m."

        autor, mensaje = rest.split(': ', 1)

        fila = {
            "fecha_raw": fecha_str,
            "hora_ampm": hora_ampm,
            "autor": autor.strip(),
            "mensaje": mensaje,
        }
        filas.append(fila)
        current = fila  # por si el siguiente renglón es continuación

    df = pd.DataFrame(filas)
    if df.empty:
        return df

    # ---- Normalizar hora + AM/PM ----
    # limpiamos espacios raros (NBSP, narrow NBSP, etc.)
    df["hora_ampm_clean"] = (
        df["hora_ampm"]
        .str.replace("\u202f", " ", regex=False)  # narrow no-break space
        .str.replace("\xa0", " ", regex=False)   # no-break space clásico
        .str.replace("\u200e", "", regex=False)  # left-to-right mark
        .str.strip()
    )

    # separar "hora" y "a. m./p. m."
    def split_hora_ampm(s):
        m = re.match(r'(\d{1,2}:\d{2}:\d{2})\s*(.*)', s)
        if not m:
            return pd.Series([None, None])
        return pd.Series(m.groups())

    df[["hora_raw", "ampm_raw"]] = df["hora_ampm_clean"].apply(split_hora_ampm)

    # normalizar a "AM"/"PM"
    df["ampm_clean"] = (
        df["ampm_raw"]
        .astype(str)
        .str.lower()
        .str.replace(".", "", regex=False)
        .str.replace(" ", "", regex=False)
        .map({"am": "AM", "pm": "PM"})
    )

    # construir timestamp completo
    df["timestamp_str"] = (
        df["fecha_raw"].astype(str) + " " +
        df["hora_raw"].astype(str) + " " +
        df["ampm_clean"].astype(str)
    )

    # convertir a datetime (12 horas + AM/PM)
    df["fecha_hora"] = pd.to_datetime(
        df["timestamp_str"],
        format="%d/%m/%y %I:%M:%S %p",
        errors="coerce",
        dayfirst=True,
    )

    # columnas derivadas
    df["fecha"] = df["fecha_hora"].dt.date
    df["mes"] = df["fecha_hora"].dt.to_period("M").astype(str)
    df["anio"] = df["fecha_hora"].dt.year
    df["longitud"] = df["mensaje"].astype(str).str.len()

    return df


# ----------------- STOPWORDS Y LIMPIEZA -----------------
STOPWORDS_ES = {
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para",
    "con","no","una","su","al","lo","como","más","pero","sus","le","ya","o",
    "fue","este","ha","sí","porque","esta","son","entre","cuando","muy","sin",
    "sobre","también","me","hasta","hay","donde","quien","desde","todo","nos",
    "durante","todos","uno","les","ni","contra","otros","ese","eso","ante",
    "ellos","e","esto","mí","antes","algunos","qué","unos","yo","otro","otras",
    "otra","él","tanto","esa","estos","mucho","quienes","nada","muchos","cual",
    "poco","ella","estar","estas","algunas","algo","nosotros","mi","mis","tú",
    "te","ti","tu","tus","ellas","nosotras","vosotros","vosotras","os",
    # palabras del contexto JL/IPAE que no aportan mucho a la nube
    "ipae","jl","jlipe","jlipa","grupo","comunidad","lideres","líderes",
    # multimedia y placeholders de whatsapp
    "omitido","omitida","imagen","sticker","stikers","foto","video","audio",
    "documento","archivo","mensaje","mensajes",
    # relleno típico
    "muchas","buenos","buenas","hola"
}

# Patrones de mensajes automáticos de sistema de WhatsApp
PATRONES_SISTEMA = [
    "se unió con el enlace del grupo",
    "se unio con el enlace del grupo",
    "creó este grupo",
    "creo este grupo",
    "añadió",
    "anadio",
    "fijó el mensaje",
    "fijo el mensaje",
    "cambió el asunto del grupo",
    "cambio el asunto del grupo",
    "cambió la foto del grupo",
    "cambio la foto del grupo",
    "los mensajes y las llamadas están cifrados",
    "los mensajes y las llamadas estan cifrados",
    "paima fijó",
    "editó",
    "aqui",
    "ustedes",
    "gran",
    "2025",
    "comparto",
    "evento",
    "cada",
    "parte",
    "solo",
    "está",
    "favor",
    "lima",
    "octubre",
    "jueves",
    "noviembre",
    "chicos",
    "eliminó",
    "espero",
    "estamos",
    "estoy",
    "mañana",
    "aquí",
    "días",
    "espero",
    "estamos",
    "numero",
    "lucero beatriz",
    "cambío número",
    "número telefónico",
    "número teléfono",
    "enviar añadir",
    "añadir nuevo",
    "nuevo número",
    "teléfono toca",
    "paima cambió",
    "gabo cade",
    "jhosely condori",
    "toda",
    "jesus costa"

]


def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", " ", texto)
    texto = re.sub(r"[^a-záéíóúüñ0-9\s]", " ", texto)
    tokens = texto.split()
    # quitamos stopwords y palabras de 3 letras o menos
    return [t for t in tokens if t not in STOPWORDS_ES and len(t) > 3]



# ----------------- CARGA DE DATA -----------------
if not DATA_PATH.exists():
    st.error(f"No se encontró el archivo {DATA_PATH}. Crea la carpeta data y pon ahí chat_jlipae.txt")
    st.stop()

with open(DATA_PATH, "r", encoding="utf-8", errors="ignore") as f:
    raw = f.read()

df = parse_whatsapp(raw)

if df.empty:
    st.error("No se lograron interpretar mensajes. Revisa que el TXT sea el exportado del chat completo.")
    st.stop()

st.success(f"Mensajes cargados: {len(df)} 🎉 - 21-12-25")
# ----------------- MAPA DE MESES BONITOS -----------------
def mes_bonito(m):
    return pd.to_datetime(m).strftime("%B %Y").capitalize()

meses = sorted(df["mes"].dropna().unique().tolist())
mes_map = {mes_bonito(m): m for m in meses}

# ----------------- CLASIFICACIÓN DE GRUPO -----------------
def normalizar(s: str) -> str:
    return str(s).strip().lower()

equipo_norm = {normalizar(a) for a in EQUIPO_IPAE_AUTORES}

df["grupo"] = df["autor"].apply(
    lambda a: "Equipo IPAE" if normalizar(a) in equipo_norm else "Miembros JL"
)

# ----------------- FILTROS -----------------
# ----------------- FILTROS -----------------
st.sidebar.header("Filtros")

# 1) Meses
with st.sidebar.expander("📅 Meses", expanded=True):
    meses_visuales = list(mes_map.keys())

    meses_vis_sel = st.multiselect(
        "Selecciona mes(es)",
        options=meses_visuales,
        default=meses_visuales
    )

    # volver a formato técnico (2025-10, etc.)
    meses_sel = [mes_map[m] for m in meses_vis_sel]

# 2) Grupo (Equipo IPAE / Miembros JL / Todos)
st.sidebar.subheader("👥 Miembros")

grupo_sel = st.sidebar.selectbox(
    "Selecciona grupo",
    ["Todos", "Equipo IPAE", "Miembros JL"],
    index=0
)

# ----------------- APLICAR FILTROS -----------------

df_base = df[df["mes"].isin(meses_sel)].copy()

if grupo_sel == "Todos":
    df_filtrado = df_base
else:
    df_filtrado = df_base[df_base["grupo"] == grupo_sel].copy()


# ----------------- RESUMEN -----------------
st.subheader("📌 Resumen general")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total mensajes", int(len(df_filtrado)))
###with c2:
###    st.metric("Personas activas", int(df_filtrado["autor"].nunique()))
with c3:
    st.metric("Meses analizados", int(df_filtrado["mes"].nunique()))

# --------- PARTICIPACIÓN DE MIEMBROS (ENGAGEMENT) ---------
st.subheader("🧩 Participación de Miembros")

total_miembros = df_filtrado["autor"].nunique()
escriben = df_filtrado[df_filtrado["mensaje"].str.len() > 3]["autor"].nunique()
silenciosos = total_miembros - escriben

# evitar división por cero
if total_miembros > 0:
    porcentaje = round((escriben / total_miembros) * 100, 1)
else:
    porcentaje = 0.0

cpm1, cpm2, cpm3, cpm4 = st.columns(4)
with cpm1:
    st.metric("Miembros totales", total_miembros)
with cpm2:
    st.metric("Participantes activos", escriben)
with cpm3:
    st.metric("Silenciosos", silenciosos)
with cpm4:
    st.metric("% Engagement", f"{porcentaje}%")

# --------- CONCENTRACIÓN DE PARTICIPACIÓN ---------
vc = df_filtrado["autor"].value_counts()   # Serie: autor -> cantidad

total_autores = int(vc.shape[0])
top_15_pct = max(1, int(total_autores * 0.15))

mensajes_top = int(vc.head(top_15_pct).sum())
mensajes_totales = int(vc.sum())

pct_concentracion = round((mensajes_top / mensajes_totales) * 100, 1) if mensajes_totales > 0 else 0.0

st.metric(
    "Concentración de participación",
    f"{pct_concentracion}%",
    help="Porcentaje de mensajes generados por el 15% de miembros más activos"
)

# ----------------- ACTIVIDAD MULTIMEDIA -----------------
st.subheader("🎨 Actividad multimedia")

lower_msgs = df_filtrado["mensaje"].astype(str).str.lower()

sticker_count = lower_msgs.str.contains("sticker").sum()
image_count   = lower_msgs.str.contains("imagen").sum()
video_count   = lower_msgs.str.contains("video").sum()
audio_count   = lower_msgs.str.contains("audio").sum()
doc_count     = lower_msgs.str.contains("documento").sum()

c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
with c_m1:
    st.metric("Stickers", int(sticker_count))
with c_m2:
    st.metric("Imágenes", int(image_count))
with c_m3:
    st.metric("Videos", int(video_count))
with c_m4:
    st.metric("Audios", int(audio_count))
with c_m5:
    st.metric("Documentos", int(doc_count))

# ----------------- EMOJIS -----------------
import emoji

st.subheader("😊 Top 10 emojis más usados")

emoji_counter = Counter()

for msg in df_filtrado["mensaje"].dropna():
    for ch in str(msg):
        if ch in emoji.EMOJI_DATA:
            emoji_counter[ch] += 1

if emoji_counter:
    top_emojis = pd.DataFrame(
        emoji_counter.most_common(10),
        columns=["emoji", "frecuencia"]
    )

    # nombre técnico del emoji (en inglés) tipo ":fire:" -> "fire"
    top_emojis["nombre"] = top_emojis["emoji"].apply(
        lambda e: emoji.demojize(e).strip(":").replace("_", " ")
    )

    # si quieres mostrar solo estas 3 columnas y centrado:
    st.dataframe(top_emojis[["emoji", "nombre", "frecuencia"]],
                 use_container_width=True)
else:
    st.info("No se detectaron emojis en los mensajes filtrados.")

# ----------------- MESES ACTIVOS -----------------
st.subheader("🗓️ Meses más activos (Top 10)")

mes_rank = (
    df_filtrado.groupby("mes")
    .size()
    .reset_index(name="mensajes")
    .sort_values("mensajes", ascending=False)
    .head(10)
)

st.dataframe(mes_rank)
# ----------------- MENSAJES POR MES -----------------
st.subheader("📅 Mensajes por mes")
mensajes_mes = (
    df_filtrado.groupby("mes")
    .size()
    .reset_index(name="mensajes")
    .sort_values("mes")
)
if not mensajes_mes.empty:
    st.bar_chart(mensajes_mes.set_index("mes"))
else:
    st.info("No hay datos filtrados por mes.")

# ----------------- DÍAS ACTIVOS -----------------
st.subheader("📆 Días más activos (Top 10)")

dia_rank = (
    df_filtrado.groupby("fecha")
    .size()
    .reset_index(name="mensajes")
    .sort_values("mensajes", ascending=False)
    .head(10)
)

st.dataframe(dia_rank)

# ----------------- HORAS ACTIVOS -----------------

st.subheader("⏰ Horas más activas (Top 10)")

df_filtrado["hora"] = df_filtrado["fecha_hora"].dt.hour

hora_rank = (
    df_filtrado.groupby("hora")
    .size()
    .reset_index(name="mensajes")
    .sort_values("mensajes", ascending=False)
    .head(10)
)

st.dataframe(hora_rank)

# ----------------- MENSAJES POR DÍA -----------------
st.subheader("📆 Mensajes por día")
mensajes_dia = (
    df_filtrado.groupby("fecha")
    .size()
    .reset_index(name="mensajes")
    .sort_values("fecha")
)
if not mensajes_dia.empty:
    st.line_chart(mensajes_dia.set_index("fecha"))
else:
    st.info("No hay datos filtrados por día.")

# ----------------- TOP HABLADORES -----------------
st.subheader("Personas que más escriben (Top 15)")
top_autores = (
    df_filtrado["autor"]
    .value_counts()
    .head(15)  # 👈 SOLO TOP 15
    .reset_index()
    .rename(columns={"index": "autor", "autor": "mensajes"})
)
st.dataframe(top_autores, use_container_width=True, height=260)




# ----------------- PALABRAS -----------------
# ----------------- PALABRAS -----------------
st.subheader("🔤 Palabras más utilizadas")

tokens_all = []
bigram_counter = Counter()

for msg in df_filtrado["mensaje"].dropna():
    msg_lower = str(msg).lower()

    # saltar mensajes de sistema (uniones, fijar, crear grupo, cifrado, etc.)
    if any(p in msg_lower for p in PATRONES_SISTEMA):
        continue

    toks = limpiar_texto(msg)
    if not toks:
        continue

    tokens_all.extend(toks)

    # bigrams (frases de dos palabras)
    for i in range(len(toks) - 1):
        bg = f"{toks[i]} {toks[i+1]}"
        bigram_counter[bg] += 1

# top palabras
word_counter = Counter(tokens_all)
top_palabras = pd.DataFrame(
    word_counter.most_common(30), columns=["palabra", "frecuencia"]
)
st.dataframe(top_palabras)

# top bigrams
st.subheader("🧩 Frases de dos palabras más usadas")
top_bigrams = pd.DataFrame(
    bigram_counter.most_common(20), columns=["frase", "frecuencia"]
)
st.dataframe(top_bigrams)


# ----------------- WORDCLOUD -----------------
st.subheader("☁️ Nube de palabras")
if tokens_all:
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(" ".join(tokens_all))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.info("No hay palabras suficientes para la nube.")



