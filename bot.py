import pandas as pd

# -------------------------------------------------------------------
# PAS 1: IMPORTACIÓ DE DADES
# -------------------------------------------------------------------

# 1. POSA AQUÍ ELS NOMS EXACTES DELS TEUS FITXERS
fitxers_atp = [
    "data/tml-data/2024.csv",
    "data/tml-data/2025.csv",
    "data/tml-data/2026.csv",
    "data/tml-data/ongoing_tourneys.csv" 
]

llista_dfs = []

# 2. Bucle per llegir i carregar cada fitxer
for fitxer in fitxers_atp:
    try:
        df = pd.read_csv(fitxer)
        llista_dfs.append(df)
        print(f"✅ {fitxer} carregat: {df.shape[0]} partits.")
    except FileNotFoundError:
        print(f"❌ ATENCIÓ: No s'ha trobat el fitxer {fitxer}. Comprova que el nom estigui bé.")

# 3. Unificació i neteja
if len(llista_dfs) > 0:
    # Unim totes les temporades
    df_atp = pd.concat(llista_dfs, ignore_index=True)
    
    # Seleccionem NOMÉS les columnes vitals pel model d'Aces 
    # (Afegim 'best_of' per saber si el partit era a 3 o 5 sets, és útil)
    columnes_vitals = [
        'tourney_date', 'surface', 'winner_name', 'loser_name', 
        'w_ace', 'w_svpt', 'l_ace', 'l_svpt', 'best_of'
    ]
    
    df_net = df_atp[columnes_vitals].copy()
    
    # Eliminem partits on no hi ha estadístiques registrades (ex: lesions o WO)
    df_net = df_net.dropna(subset=['w_ace', 'l_ace', 'w_svpt', 'l_svpt'])
    
    # Ordenem cronològicament (Clau per no veure el futur més endavant)
    df_net = df_net.sort_values(by=['tourney_date']).reset_index(drop=True)
    
    print("-" * 50)
    print(f"MACRO-DATASET ATP COMPLETAT: {df_net.shape[0]} partits totalment vàlids.")
    print("-" * 50)
    display(df_net.head())

# -------------------------------------------------------------------
# PAS 2: TRANSFORMACIÓ I CÀLCUL DE MÈTRIQUES BASE
# -------------------------------------------------------------------

# 1. Creem la visió des de la perspectiva dels GUANYADORS
df_winners = df_net[['tourney_date', 'surface', 'winner_name', 'loser_name', 'w_ace', 'w_svpt', 'l_ace', 'l_svpt', 'best_of']].copy()
# Reanomenem les columnes de forma neutral
df_winners.columns = ['Date', 'Surface', 'Player', 'Opponent', 'Aces_Made', 'Svpt_Made', 'Aces_Received', 'Svpt_Opponent', 'Best_of']

# 2. Creem la visió des de la perspectiva dels PERDEDORS
df_losers = df_net[['tourney_date', 'surface', 'loser_name', 'winner_name', 'l_ace', 'l_svpt', 'w_ace', 'w_svpt', 'best_of']].copy()
# Reanomenem invertint l'ordre: els Aces_Made del perdedor eren 'l_ace', i els Aces_Received eren 'w_ace'
df_losers.columns = ['Date', 'Surface', 'Player', 'Opponent', 'Aces_Made', 'Svpt_Made', 'Aces_Received', 'Svpt_Opponent', 'Best_of']

# 3. Unim les dues meitats (Stacking)
df_jugadors = pd.concat([df_winners, df_losers], ignore_index=True)

# 4. Ordenem per data per mantenir la integritat cronològica (Vital per al ML)
df_jugadors = df_jugadors.sort_values(by=['Date']).reset_index(drop=True)

# 5. CALCULEM ELS RATIOS (La veritat matemàtica)
# .replace(0, 1) evita un error fatal matemàtic si algun partit té registrat 0 punts de servei per error
df_jugadors['Ace_Rate'] = df_jugadors['Aces_Made'] / df_jugadors['Svpt_Made'].replace(0, 1)
df_jugadors['Ace_Allowed_Rate'] = df_jugadors['Aces_Received'] / df_jugadors['Svpt_Opponent'].replace(0, 1)

print(f"Transformació completada! Hem passat de {df_net.shape[0]} partits a {df_jugadors.shape[0]} registres de jugadors.")
display(df_jugadors.head(10))

# -------------------------------------------------------------------
# PAS 3: ENGINYERIA DE CARACTERÍSTIQUES (Mitjanes Exponencials)
# -------------------------------------------------------------------

# 1. FORMAT DE LA DATA
# A la imatge veiem que la data està com '20241229' (un simple número). 
# Ho convertim a format Date real per assegurar l'ordre cronològic absolut.
df_jugadors['Date'] = pd.to_datetime(df_jugadors['Date'].astype(str), format='%Y%m%d', errors='coerce')
df_jugadors = df_jugadors.sort_values(by=['Date']).reset_index(drop=True)

# 2. DEFINIM L'ABAST DE LA MITJANA EXPONENCIAL
# span=15 significa que la mitjana "recorda" tot l'històric, però dóna gairebé tot 
# el pes matemàtic als últims 15 partits (l'estat de forma actual).
finestra_ema = 15

# 3. CÀLCUL DE LA FORÇA DE SERVEI I REST (Amb desplaçament temporal per no veure el futur)
# Calculem quin Ace Rate portava el jugador FINS AQUELL DIA
df_jugadors['Exp_Ace_Rate'] = df_jugadors.groupby('Player')['Ace_Rate'].transform(
    lambda x: x.ewm(span=finestra_ema, min_periods=1).mean().shift(1)
)

# Calculem quin Ace Allowed Rate (Aces que es deixa fer) portava l'Oponent FINS AQUELL DIA
# Atenció aquí: ho agrupem per 'Opponent'
df_jugadors['Exp_Opponent_Allowed_Rate'] = df_jugadors.groupby('Opponent')['Ace_Allowed_Rate'].transform(
    lambda x: x.ewm(span=finestra_ema, min_periods=1).mean().shift(1)
)

# 4. ELIMINEM NULS
# Els primers partits on un jugador encara no té històric sortiran nuls, els netegem.
df_model_aces = df_jugadors.dropna(subset=['Exp_Ace_Rate', 'Exp_Opponent_Allowed_Rate']).copy()

# 5. CREEM LA PREVISIÓ (La nostra "Pista" estrella)
# Si Alcaraz sol fer un 15% d'Aces, i Medvedev sol permetre un 12% d'Aces... quina és la previsió?
# Creuem les dades multiplicant l'habilitat del jugador per la feblesa de l'oponent.
df_model_aces['Projected_Ace_Rate'] = (df_model_aces['Exp_Ace_Rate'] + df_model_aces['Exp_Opponent_Allowed_Rate']) / 2

# Imprimim el resultat per a un jugador conegut per veure si té sentit
print(f"Base de dades enriquida: {df_model_aces.shape[0]} registres preparats pel Machine Learning.")

# Anem a veure com preveu el model un canoner com Taylor Fritz
columnes_mostrar = ['Date', 'Player', 'Opponent', 'Surface', 'Exp_Ace_Rate', 'Exp_Opponent_Allowed_Rate', 'Projected_Ace_Rate', 'Aces_Made']
display(df_model_aces[df_model_aces['Player'] == 'Taylor Fritz'][columnes_mostrar].tail(10))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# PAS 4 CORREGIT: EL CREUAMENT PERFECTE (SELF-MERGE)
# -------------------------------------------------------------------
df_temp = df_model_aces.copy()

# 1. Calculem l'històric SEMPRE des de la perspectiva del 'Player' (Sense barrejar!)
df_temp['P_Ace_Rate'] = df_temp.groupby(['Player', 'Surface'])['Ace_Rate'].transform(
    lambda x: x.ewm(span=15, min_periods=1).mean().shift(1)
)
df_temp['P_Allowed_Rate'] = df_temp.groupby(['Player', 'Surface'])['Ace_Allowed_Rate'].transform(
    lambda x: x.ewm(span=15, min_periods=1).mean().shift(1)
)
df_temp['P_Svpt'] = df_temp.groupby(['Player', 'Surface'])['Svpt_Made'].transform(
    lambda x: x.ewm(span=15, min_periods=1).mean().shift(1)
)

# 2. Resolem el problema de l'Oponent: Creuem la taula amb si mateixa
# Agafem només la debilitat al rest de tothom
df_oponent = df_temp[['Date', 'Player', 'Opponent', 'P_Allowed_Rate']].copy()
# Intercanviem els noms per poder fer el "Match"
df_oponent.columns = ['Date', 'Opponent', 'Player', 'Opponent_Allowed_Rate']

# Unim les dades: Ara cada 'Player' té al costat la debilitat REAL del seu 'Opponent' d'aquell dia
df_ml = pd.merge(df_temp, df_oponent, on=['Date', 'Player', 'Opponent'], how='left')

# 3. Neteja i Projecció final
df_ml = df_ml.dropna(subset=['P_Ace_Rate', 'Opponent_Allowed_Rate', 'P_Svpt', 'Aces_Made']).copy()
df_ml['Projected_Ace_Rate'] = (df_ml['P_Ace_Rate'] + df_ml['Opponent_Allowed_Rate']) / 2

# 4. Entrenament de la IA
encoder_surface = LabelEncoder()
df_ml['Surface_Encoded'] = encoder_surface.fit_transform(df_ml['Surface'])

columnes_features = ['Projected_Ace_Rate', 'P_Svpt', 'Best_of', 'Surface_Encoded']
X = df_ml[columnes_features]
y = df_ml['Aces_Made']

print("Entrenant l'Oracle Definitiu (Sense barreja de dades)...")
model_rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
model_rf.fit(X, y)
print("✅ Oracle operatiu i calibrat!\n")

# -------------------------------------------------------------------
# PAS 5 CORREGIT: L'ORACLE LÒGIC
# -------------------------------------------------------------------
def predir_aces_definitiu(jugador1, jugador2, superficie='Hard', best_of=3):
    df_surf = df_ml[df_ml['Surface'] == superficie]
    
    # Ara buscar les estadístiques és super fàcil: només mirem on són 'Player'
    j1_stats = df_surf[df_surf['Player'] == jugador1]
    j2_stats = df_surf[df_surf['Player'] == jugador2]
    
    if j1_stats.empty or j2_stats.empty:
        return print(f"Error: Falten dades d'algun jugador en pista '{superficie}'.")

    # Extraiem l'històric del Jugador 1
    j1_serve = j1_stats.iloc[-1]['P_Ace_Rate']
    j1_return = j1_stats.iloc[-1]['P_Allowed_Rate']
    j1_svpt = j1_stats.iloc[-1]['P_Svpt']
    
    # Extraiem l'històric del Jugador 2
    j2_serve = j2_stats.iloc[-1]['P_Ace_Rate']
    j2_return = j2_stats.iloc[-1]['P_Allowed_Rate']
    j2_svpt = j2_stats.iloc[-1]['P_Svpt']
    
    # Creuem el servei d'un amb el rest de l'altre
    j1_proj = (j1_serve + j2_return) / 2
    j2_proj = (j2_serve + j1_return) / 2
    
    surf_encoded = encoder_surface.transform([superficie])[0]
    
    # Construïm la matriu per la IA
    partit_j1 = pd.DataFrame({'Projected_Ace_Rate': [j1_proj], 'P_Svpt': [j1_svpt], 'Best_of': [best_of], 'Surface_Encoded': [surf_encoded]})
    partit_j2 = pd.DataFrame({'Projected_Ace_Rate': [j2_proj], 'P_Svpt': [j2_svpt], 'Best_of': [best_of], 'Surface_Encoded': [surf_encoded]})
    
    aces_j1 = model_rf.predict(partit_j1)[0]
    aces_j2 = model_rf.predict(partit_j2)[0]
    
    print("-" * 50)
    print(f"🎾 ORACLE PREDICTIU: {jugador1} vs {jugador2}")
    print(f"🌍 Pista: {superficie}  |  ⚙️ Format: Millor de {best_of} sets")
    print("-" * 50)
    print(f"👉 Predicció {jugador1}: {aces_j1:.1f} Aces")
    print(f"👉 Predicció {jugador2}: {aces_j2:.1f} Aces")
    print("-" * 50)

# PROVES: Ara SÍ que veurem la diferència real entre un canoner i la resta
predir_aces_definitiu("Hubert Hurkacz", "Carlos Alcaraz", superficie="Hard", best_of=3)
predir_aces_definitiu("Hubert Hurkacz", "Carlos Alcaraz", superficie="Clay", best_of=3)
predir_aces_definitiu("Hubert Hurkacz", "Carlos Alcaraz", superficie="Grass", best_of=5)

# -------------------------------------------------------------------
# ORACLE TESTER
# -------------------------------------------------------------------

import telebot

# 1. CONNECTEM AMB TELEGRAM (Substitueix per la teva clau real)
TOKEN = "8979683932:AAHm-5YInca_vRSs6jewLFQIwFTo8BTtaHQ"
bot = telebot.TeleBot(TOKEN)

# 2. ADAPTEM L'ORACLE PERQUÈ RETORNI TEXT EN LLOC DE PRINT
def oracle_telegram(jugador1, jugador2, superficie='Hard', best_of=3):
    df_surf = df_ml[df_ml['Surface'] == superficie]
    
    j1_stats = df_surf[df_surf['Player'] == jugador1.strip()]
    j2_stats = df_surf[df_surf['Player'] == jugador2.strip()]
    
    if j1_stats.empty or j2_stats.empty:
        return f"⚠️ Error: Falten dades d'algun jugador en pista '{superficie}'. Comprova que els noms estiguin escrits exactament com a l'ATP."

    # Extraiem l'històric (Servei, Rest i Volum)
    j1_serve = j1_stats.iloc[-1]['P_Ace_Rate']
    j1_return = j1_stats.iloc[-1]['P_Allowed_Rate']
    j1_svpt = j1_stats.iloc[-1]['P_Svpt']
    
    j2_serve = j2_stats.iloc[-1]['P_Ace_Rate']
    j2_return = j2_stats.iloc[-1]['P_Allowed_Rate']
    j2_svpt = j2_stats.iloc[-1]['P_Svpt']
    
    # Projeccions
    j1_proj = (j1_serve + j2_return) / 2
    j2_proj = (j2_serve + j1_return) / 2
    
    surf_encoded = encoder_surface.transform([superficie])[0]
    
    partit_j1 = pd.DataFrame({'Projected_Ace_Rate': [j1_proj], 'P_Svpt': [j1_svpt], 'Best_of': [best_of], 'Surface_Encoded': [surf_encoded]})
    partit_j2 = pd.DataFrame({'Projected_Ace_Rate': [j2_proj], 'P_Svpt': [j2_svpt], 'Best_of': [best_of], 'Surface_Encoded': [surf_encoded]})
    
    aces_j1 = model_rf.predict(partit_j1)[0]
    aces_j2 = model_rf.predict(partit_j2)[0]
    
    # Construïm el missatge visual per a Telegram
    missatge = (
        f"🎾 *ORACLE PREDICTIU*\n"
        f"👤 {jugador1.strip()} vs {jugador2.strip()}\n"
        f"🌍 Pista: {superficie} | ⚙️ Sets: Millor de {best_of}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"👉 Predicció {jugador1.strip()}: *{aces_j1:.1f} Aces*\n"
        f"👉 Predicció {jugador2.strip()}: *{aces_j2:.1f} Aces*"
    )
    return missatge

# 3. CREEM ELS COMANDAMENTS DEL BOT
@bot.message_handler(commands=['start', 'help'])
def enviar_benvinguda(message):
    text = (
        "Hola! Soc el teu Oracle de Tennis basat en Machine Learning 🤖🎾\n\n"
        "Per demanar-me una predicció, fes servir la comanda /aces seguida dels paràmetres separats per comes:\n\n"
        "👉 `/aces Jugador 1, Jugador 2, Superfície, Sets`\n\n"
        "📌 *Exemple:*\n"
        "`/aces Hubert Hurkacz, Carlos Alcaraz, Hard, 3`"
    )
    bot.reply_to(message, text, parse_mode='Markdown')

@bot.message_handler(commands=['aces'])
def predir_comanda(message):
    try:
        # Extraiem el text que va després de "/aces "
        arguments = message.text.replace('/aces ', '').split(',')
        
        if len(arguments) != 4:
            bot.reply_to(message, "⚠️ Format incorrecte. Recorda separar els 4 paràmetres per comes.\nExemple: `/aces Jannik Sinner, Carlos Alcaraz, Hard, 3`", parse_mode='Markdown')
            return
            
        # Netejem els espais en blanc que l'usuari hagi pogut posar sense voler
        j1 = arguments[0].strip()
        j2 = arguments[1].strip()
        surf = arguments[2].strip()
        sets = int(arguments[3].strip())
        
        # Cridem a l'Oracle i enviem la resposta
        resposta = oracle_telegram(j1, j2, surf, sets)
        bot.reply_to(message, resposta, parse_mode='Markdown')
        
    except Exception as e:
        bot.reply_to(message, f"❌ Hi ha hagut un error intern o el format no és vàlid. Revisa els noms.")

# 4. ENGEGUEM EL BOT (Es quedarà escoltant permanentment)
print("🤖 Bot de Telegram iniciat i escoltant missatges...")
bot.infinity_polling()