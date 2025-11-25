from fastapi import FastAPI
from pydantic import BaseModel
from langchain_neo4j import Neo4jGraph

# from langchain_ollama import OllamaLLM
# Correggi la riga 6 (o quella dove si trova l'import)
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from cachetools import TTLCache
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pydantic import BaseModel
import re
import logging
import time

# Importazione dei modelli LLM
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import os


MODELS = {
    "gpt-4o": ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key="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",
    ),
    "gpt-4o-mini": ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key="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",
    ),
    "llama3": ChatOllama(
        model="llama3",  # Il modello che gira sul server remoto
        temperature=0,
        base_url="http://172.16.30.23:11434",
        request_timeout=120.0,  #
    ),
    "mistral": ChatOllama(
        model="mistral",  # Se vuoi testare anche mistral su quel server
        temperature=0,
        base_url="http://172.16.30.23:11434",
        request_timeout=120.0,
    ),
    "gemini-2.5-flash": ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_output_tokens=8000,
        api_key="AIzaSyDrZGR3fth5wq1eR1WjFfKezz0e20i9eDs",
    ),
}

# 2. Scelta del modello
MODELLO_SELEZIONATO = "llama3"  # Cambia qui per selezionare il modello desiderato

# 3. Inizializza l'LLM scelto
llm = MODELS[MODELLO_SELEZIONATO]

print(f"✅ Modello LLM in uso: {MODELLO_SELEZIONATO}")


# Configurazione logging per debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. CONFIGURAZIONE E CACHING ---

app = FastAPI()
# Cache per query Cypher (scade dopo 10 minuti)
query_cache = TTLCache(maxsize=100, ttl=600)
# Configura il modello LLM locale
# llm = OllamaLLM(model="llama3:8b")

# Configura la connessione al grafo
graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="Blue2018")


# Definizione degli schemi per le richieste e le risposte
class ConversationMessage(BaseModel):
    timestamp: datetime
    question: str
    answer: str
    context: List[Dict] = []
    query_generated: str = ""


class ConversationSession(BaseModel):
    user_id: str
    messages: List[ConversationMessage] = []
    created_at: datetime
    last_activity: datetime


# Cache globale per le sessioni di conversazione (scade dopo 30 minuti di inattività)
conversation_sessions: Dict[str, ConversationSession] = {}
SESSION_TIMEOUT = timedelta(minutes=30)


def get_or_create_session(user_id: str) -> ConversationSession:
    """Ottieni o crea una sessione di conversazione per l'utente"""
    now = datetime.now()

    # Pulisci sessioni scadute
    expired_sessions = [
        uid
        for uid, session in conversation_sessions.items()
        if now - session.last_activity > SESSION_TIMEOUT
    ]
    for uid in expired_sessions:
        del conversation_sessions[uid]
        logger.info(f" Sessione scaduta eliminata per utente: {uid}")

    # Ottieni o crea sessione
    if user_id not in conversation_sessions:
        conversation_sessions[user_id] = ConversationSession(
            user_id=user_id, created_at=now, last_activity=now
        )
        logger.info(f" Nuova sessione creata per utente: {user_id}")
    else:
        conversation_sessions[user_id].last_activity = now

    return conversation_sessions[user_id]


def add_message_to_session(
    user_id: str, question: str, answer: str, context: List[Dict], query: str
):
    """Aggiungi un messaggio alla sessione di conversazione"""
    session = get_or_create_session(user_id)

    message = ConversationMessage(
        timestamp=datetime.now(),
        question=question,
        answer=answer,
        context=context,
        query_generated=query,
    )

    session.messages.append(message)

    # Mantieni solo gli ultimi 5 messaggi per evitare memory overflow
    if len(session.messages) > 15:
        session.messages = session.messages[-15:]

    logger.info(
        f" Messaggio aggiunto alla sessione. Totale messaggi: {len(session.messages)}"
    )


def get_conversation_context(user_id: str) -> str:
    """Genera contesto conversazione COMPATTO"""
    session = conversation_sessions.get(user_id)
    if not session or len(session.messages) == 0:
        return ""

    last_msg = session.messages[-1]  # SOLO ultimo messaggio

    context_lines = ["CONVERSAZIONE PRECEDENTE (ultimo messaggio):"]
    context_lines.append(f"Domanda: {last_msg.question}")

    # Estrai SOLO nomi, niente altro
    if last_msg.context:
        nomi_clienti = []
        nomi_fornitori = []

        for item in last_msg.context:
            if "cliente" in item and item["cliente"]:
                nomi_clienti.append(item["cliente"])
            elif "fornitore" in item and item["fornitore"]:
                nomi_fornitori.append(item["fornitore"])

        if nomi_clienti:
            # Lista Python per facilitare parsing LLM
            context_lines.append(f"LISTA_CLIENTI = {nomi_clienti}")
        if nomi_fornitori:
            context_lines.append(f"LISTA_FORNITORI = {nomi_fornitori}")

    return "\n".join(context_lines)


def classify_intent(question: str) -> str:
    """Classifica l'intento della domanda"""
    INTENT_PROMPT = f"""Classifica la domanda in UNA categoria:
    - DATABASE_QUERY: Richiede dati (fatturato, clienti, fornitori, liste)
    - CONVERSATIONAL: Saluti, ringraziamenti ("ciao", "grazie")
    - CLARIFICATION: Chiede spiegazioni su risposta precedente
    - SYSTEM_HELP: Chiede cosa può fare il sistema

    Domanda: "{question}"
    Rispondi SOLO con il nome della categoria."""

    try:
        response = llm.invoke(INTENT_PROMPT)
        intent = response.content.strip().upper()
        if intent in [
            "DATABASE_QUERY",
            "CONVERSATIONAL",
            "CLARIFICATION",
            "SYSTEM_HELP",
        ]:
            return intent
        return "DATABASE_QUERY"
    except:
        return "DATABASE_QUERY"


def correggi_nomi_fuzzy_neo4j(question: str) -> tuple[str, list]:
    """Usa full-text search Neo4j. Scala su milioni di record."""
    correzioni = []

    # BLACKLIST: parole comuni da NON correggere MAI
    BLACKLIST = {
        "mostra",
        "dammi",
        "elenca",
        "trova",
        "cerca",
        "qual",
        "quale",
        "quali",
        "tutti",
        "tutte",
        "tutto",
        "ogni",
        "totale",
        "somma",
        "conta",
        "quanti",
        "clienti",
        "cliente",
        "fornitori",
        "fornitore",
        "articoli",
        "articolo",
        "ditte",
        "ditta",
        "fatture",
        "fattura",
        "documenti",
        "documento",
        "anno",
        "mese",
        "giorno",
        "data",
        "importo",
        "fatturato",
        "costo",
        "dove",
        "come",
        "quando",
        "perché",
        "cosa",
        "chi",
        "nel",
        "del",
        "della",
        "delle",
        "dei",
        "con",
        "per",
        "tra",
    }

    parole = [p for p in question.split() if len(p) >= 4]  # Min 4 caratteri

    indici = [
        ("clienti_fuzzy", "Cliente", "name"),
        ("fornitori_fuzzy", "GruppoFornitore", "ragioneSociale"),
        ("articoli_fuzzy", "Articolo", "descrizione"),
        ("doctype_fuzzy", "DocType", "name"),
        ("luoghi_fuzzy", "Luogo", "localita"),
    ]

    for parola in parole:
        # SKIP se è parola comune
        if parola.lower() in BLACKLIST:
            continue

        best_match = None
        best_score = 0
        best_tipo = None

        for idx_name, label, prop in indici:
            try:
                # Solo 1 edit distance (~1 carattere sbagliato)
                query = f"""
                CALL db.index.fulltext.queryNodes('{idx_name}', '{parola}~1') 
                YIELD node, score
                RETURN node.{prop} as nome, score
                ORDER BY score DESC
                LIMIT 1
                """
                result = graph.query(query)

                if result and result[0]["score"] > best_score:
                    best_match = result[0]["nome"]
                    best_score = result[0]["score"]
                    best_tipo = label
            except:
                continue

        # Applica SOLO se score MOLTO alto (>2.0) e nome realmente simile
        if best_match and best_score > 0.8:
            # Ulteriore check: almeno 70% di caratteri in comune
            similarity = len(set(parola.lower()) & set(best_match.lower())) / max(
                len(parola), len(best_match)
            )
            if similarity > 0.7 and parola.lower() != best_match.lower():
                question = question.replace(parola, best_match)
                correzioni.append(
                    f"'{parola}' → '{best_match}' ({best_tipo}, score={best_score:.2f})"
                )

    return question, correzioni


# OTTIMIZZAZIONE 1: Caricamento schema grafo in memoria all'avvio
print("Caricamento dello schema del grafo in memoria...")
try:
    graph.refresh_schema()
    print("✅ Schema caricato.")
    logger.info(f"Schema caricato: {graph.schema}")
except Exception as e:
    print(
        f" ERRORE: Impossibile caricare lo schema. Assicurati che Neo4j sia attivo e APOC installato. Dettagli: {e}"
    )
    logger.error(f"Errore caricamento schema: {e}")

# --- 2. TEMPLATE E FUNZIONI AUSILIARIE ---


def format_number_italian(number):
    """Formatta un numero in stile italiano: 1.234,56"""
    if isinstance(number, (int, float)):
        return f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return str(number)


def create_fallback_response(context, question):
    """
    Crea una risposta di fallback intelligente, capace di formattare
    qualsiasi tipo di lista semplice.
    """
    if not context or len(context) == 0:
        return "Non ho trovato dati corrispondenti alla tua richiesta nel database."

    total_results = len(context)
    first_item = context[0]
    # --- Logica per singolo dizionario (es. totale fatturato) ---
    if total_results == 1 and isinstance(first_item, dict):
        response_parts = ["Ho trovato il seguente risultato:"]
        for key, value in first_item.items():
            formatted_key = key.replace("_", " ").capitalize()
            formatted_value = (
                format_number_italian(value)
                if isinstance(value, (int, float))
                else value
            )
            response_parts.append(f"- **{formatted_key}**: {formatted_value}")
        return "\n".join(response_parts)

    # --- Logica per le liste (ora gestisce anche il caso in cui l'AI fallisce) ---
    if isinstance(first_item, dict) and len(first_item.keys()) == 1:
        key_name = list(first_item.keys())[0]
        intro_sentence = f"Ho trovato {total_results} risultati:"
        items_list = [f"- {item.get(key_name, 'N/D')}" for item in context]
        return "\n".join([intro_sentence] + items_list)

    # --- VECCHIA LOGICA (mantenuta per casi più complessi, come i totali) ---
    elif any(key in ["fatturato", "importo", "total"] for key in first_item.keys()):
        for key in ["fatturato", "importo", "total"]:
            if key in first_item:
                value = first_item[key]
                if isinstance(value, (int, float)):
                    formatted_value = format_number_italian(value)
                    return f"Il risultato del calcolo è di {formatted_value} €"

    # --- FALLBACK FINALE (se la struttura dei dati è sconosciuta) ---
    return f"Ho trovato {total_results} risultati, ma non riesco a formattarli in modo leggibile."


# Template unificato per generare query Cypher con contesto conversazione
CYPHER_GENERATION_TEMPLATE = """
Task: Sei un esperto sviluppatore Neo4j. Il tuo unico compito è generare una singola e precisa query Cypher per rispondere alla domanda di un utente, basandoti sullo schema del grafo, le regole e gli esempi forniti.

IMPORTANTE: Potresti essere nel mezzo di una conversazione. Se la sezione "Conversazione Precedente" qui sotto non è vuota, usala per capire il contesto, risolvere riferimenti ("questo cliente") o modificare parametri ("e nel 2022?"). Se è vuota, ignora questa parte.

---
**CONVERSAZIONE PRECEDENTE:**
{conversation_context}
---

**LOGICA DI MEMORIA (DA APPLICARE SE LA CONVERSAZIONE NON È VUOTA):**

1.  **Riferimento a Risultato dalla Risposta**: Se la Domanda Attuale usa pronomi come "di questa famiglia", "su questo cliente", "di questo prodotto", estrai il nome dell'entità dalla *risposta precedente* e usalo nella clausola `WHERE`.
2.  **Modifica di Parametro**: Se la Domanda Attuale cambia solo un dettaglio di una query precedente (es. "e nel 2022?" dopo una domanda sul 2024), modifica solo quel parametro.
3.  **Azione su una Lista**: Se la conversazione contiene `LISTA_CLIENTI` o `LISTA_FORNITORI` e la domanda usa parole come "loro", "questi", "di tutti", usa la clausola `WHERE ... IN [lista]`.
4.  **Analisi di Drill-Down**: Se la risposta precedente era un totale aggregato e la domanda attuale chiede un'analisi più dettagliata (es. "mostrami il dettaglio per singolo fornitore"), crea una nuova query che raggruppi i dati come richiesto.

**Esempi Specifici di Ragionamento con Memoria:**

* **Esempio di Riferimento a Risultato:**
    * *Conversazione Precedente*: Risposta: "La famiglia più venduta è CENTRALI FRIGO..."
    * *Domanda Attuale*: "Ok, quali sono i prodotti di questa famiglia?"
    * *Query Generata*:
        ```cypher
        MATCH (fam:Famiglia)<-[:INCLUSA_IN]-(:Sottofamiglia)<-[:APPARTIENE_A]-(a:Articolo)
        WHERE toLower(trim(fam.nome)) = 'centrali frigo'
        RETURN a.descrizione AS prodotto
        ```

* **Esempio di Azione su Lista:**
    * *Conversazione Precedente*: `LISTA_CLIENTI = ['ACME SRL', 'ROSSI SPA']`
    * *Domanda Attuale*: "dammi il loro fatturato"
    * *Query Generata*:
        ```cypher
        MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)
        WHERE toLower(trim(c.name)) IN ['acme srl', 'rossi spa'] AND dr.tipoValore = 'Fatturato' AND dr.importo > 0
        RETURN c.name AS cliente, sum(dr.importo) AS fatturato
        ```
        
* **Esempio: Analisi di Drill-Down**
* **Conversazione Precedente:**
    * Domanda: "Qual è il totale acquistato da tutti i fornitori?"
    * Risposta: "Il totale acquistato è di 5.000.000 €"
* **Domanda Attuale:** "Sì, voglio approfondire analizzando il totale per singolo fornitore"
* **Ragionamento Interno:** L'utente vuole scomporre il totale precedente. Devo raggruppare per fornitore e sommare gli importi per ciascuno. In Cypher, il raggruppamento è implicito nel RETURN.
* **Query Cypher Generata:**
    ```cypher
    MATCH (gf:GruppoFornitore)<-[:RAGGRUPPATO_SOTTO]-(:Fornitore)-[:HA_EMESSO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)
    RETURN gf.ragioneSociale AS fornitore, sum(dr.importoNetto) AS totaleAcquistato
    ORDER BY totaleAcquistato DESC
    ```
    
**Esempi di Ragionamento con Memoria:**
* **Esempio Riferimento a Entità dalla Domanda**
    * *Conversazione Precedente*: Domanda: "Documenti del cliente con codice 1-C-3241?"
    * *Domanda Attuale*: "Qual è il nome di questo cliente?"
    * *Query Cypher*:
        ```cypher
        MATCH (c:Cliente {{accountnumber: '1-C-3241'}}) RETURN c.name
        ```

**REGOLE FONDAMENTALI:**
1.  **Sicurezza**: Usa SOLO `MATCH` e `RETURN`.
2.  **Robustezza**:
    * Per filtrare su **ID esatti** (`dittaId`, `accountnumber`, `codice`), usa il match diretto: `MATCH (n {{proprieta: 'valore'}})`.
    * Per filtrare su **stringhe di testo** (`name`, `descrizione`, `localita`), usa SEMPRE `WHERE toLower(trim(var.prop)) = 'valore minuscolo'`.
3.  **Output Specifico**: Quando l'utente chiede una lista di entità (es. "mostrami le ditte"), restituisci la loro proprietà identificativa (`d.dittaId`, `c.name`), non l'intero nodo e ordina alfabeticamente con `ORDER BY`.
4.  **PATTERN DI CODICE OBBLIGATORI PER I CALCOLI**:
    * **SE la domanda contiene "fatturato", "venduto" o "incassato"**:
        La clausola `WHERE` DEVE OBBLIGATORIAMENTE contenere questo blocco di codice ESATTO:
        ```cypher
        WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0
        ```
    * **SE la domanda contiene "costo" o "acquistato"**:
        Il percorso DEVE iniziare da `(:Fornitore)-[:HA_EMESSO]->...` e la somma deve essere su `dr.importo`. Non aggiungere altri filtri sull'importo a meno che non siano richiesti.
    * **SE la domanda chiede un "valore totale" generico (es. "valore movimentato")**:
        Usa la clausola `WHERE dr.importo > 0` ma NON usare il filtro `dr.tipoValore`.        
5.  **Distinzione Quantità vs. Valore**: Per "più popolare" o "più acquistato/venduto" in termini di numero, usa `sum(dr.quantita)`. Per "più valore" o "più importante", usa `sum(dr.importo)`.
6.  **Distinzione Cliente vs. Fornitore (REGOLA CRITICA E OBBLIGATORIA)**:
    * Qualsiasi calcolo di **"VENDITE", "FATTURATO" o "POPOLARITÀ"** DEVE OBBLIGATORIAMENTE iniziare il percorso da `(:Cliente)-[:HA_RICEVUTO]->...`.
    * I **"COSTI/ACQUISTI"** DEVONO iniziare da `(:Fornitore)-[:HA_EMESSO]->...`.
7.  **Date**: Usa `.year`, `.month`, `.day` sulla proprietà `dataEmissione`.
8.  **STRUTTURA DELLA QUERY: `MATCH` vs. `WITH` (REGOLA FONDAMENTALE)**
    * **Priorità 1 (Caso Standard):** La tua priorità assoluta è usare un **singolo `MATCH`** per descrivere un percorso logico e continuo.
    * **Priorità 2 (Casi Speciali):** Usa `WITH` **UNICAMENTE** in questi due scenari:
        * **A) Per Analisi a Cascata**: Quando devi passare una **lista di risultati** da una fase di analisi alla successiva.
        * **B) Per Filtrare su un'Aggregazione**: Quando devi filtrare i risultati **dopo** aver calcolato una somma o un conteggio.
9.  **CLASSIFICHE "TOP N"**: Per domande che cercano "il migliore", raggruppa sempre per la **proprietà testuale** (`c.name`, `gf.ragioneSociale`), non per il nodo intero, e usa `RETURN ... ORDER BY ... LIMIT N`.
10. **Ordinamento Risultati (NUOVA REGOLA)**: Quando calcoli aggregazioni (somme, conteggi), ordina sempre i risultati in modo decrescente per il valore calcolato, a meno che la domanda non chieda diversamente.
11. **LISTE UNICHE**: Usa SEMPRE `DISTINCT` quando elenchi entità per evitare duplicati.
12. **QUERY NEGATIVE**: Per domande che cercano "nessuno", "mai", "non ha", usa metodo corretto (`OPTIONAL MATCH ... WHERE ... IS NULL`).
14. **DICHIARAZIONE DELLE VARIABILI (ERRORE COMUNE DA EVITARE)**:
    Ogni variabile usata in `WHERE` o `RETURN` DEVE essere definita nel `MATCH` con un alias. Presta la massima attenzione a questo punto, è un errore grave.
    * **ESEMPIO ERRATO**: `MATCH (:Documento)-... WHERE doc.dataEmissione.year = 2024` (ERRORE: `doc` non è definito).
    * **ESEMPIO CORRETTO**: `MATCH (doc:Documento)-... WHERE doc.dataEmissione.year = 2024` (CORRETTO: `doc` è definito).
15. **PROPRIETÀ NODO DITTA**: Usa `d.dittaId` per identificare una ditta, MAI `d.name` o `d.nome` o `d.ragioneSociale`.
16. **Output**: SOLO la query Cypher, senza spiegazioni.
17. Se ti chiedono tipo la provincia, rispondi con la sigla (es. PG per Perugia). Se non la sai, usa lcoalita , quindi l.localita(es. Perugia).
**Schema del Grafo (UFFICIALE E CORRETTO):**
* Nodi: `(:Ditta)`, `(:Cliente)`, `(:GruppoFornitore)`, `(:Fornitore)`, `(:Articolo)`, `(:Documento)`, `(:RigaDocumento)`, `(:Luogo)`, `(:DocType)`, `(:Famiglia)`, `(:Sottofamiglia)`.
* Relazioni:
  * (:Fornitore)-[:RAGGRUPPATO_SOTTO]->(:GruppoFornitore)
  * (:Fornitore)-[:APPARTIENE_A]->(:Ditta)
  * (:Fornitore)-[:SI_TROVA_A]->(:Luogo)
  * (:Fornitore)-[:HA_EMESSO]->(:Documento)
  * (:Cliente)-[:APPARTIENE_A]->(:Ditta)
  * (:Cliente)-[:HAS_ADDRESS]->(:Luogo)
  * (:Cliente)-[:HA_RICEVUTO]->(:Documento)
  * (:Articolo)-[:APPARTIENE_A]->(:Sottofamiglia)
  * (:Sottofamiglia)-[:INCLUSA_IN]->(:Famiglia)
  * (:Documento)-[:IS_TYPE]->(:DocType)
  * (:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)
  * (:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(:Articolo)

**Esempi di Query (Impara da questi pattern CORRETTI):**

// --- ESEMPI DI CALCOLO E AGGREGAZIONE ---

* **Domanda**: "Dimmi il fatturato di ogni ditta."
* **Query**:
  ```cypher
  MATCH (d:Ditta)<-[:APPARTIENE_A]-(:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)
  WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 
  RETURN d.dittaId AS ditta, sum(dr.importo) AS fatturatoTotale ORDER BY fatturatoTotale DESC
    ```

* **Domanda**: "Qual è il fatturato totale del cliente BARONE DI FERRANTE EZIO?"
* **Query**:
  ```cypher
  MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)
  WHERE toLower(trim(c.name)) = 'barone di ferrante ezio' AND dr.tipoValore = 'Fatturato' AND dr.importo > 0
  RETURN sum(dr.importo) AS fatturatoTotale
    ```
* **Domanda**: "Cliente con maggior fatturato?"
* **Query**:
    ```cypher
    MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)
    WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0
    RETURN c.name AS cliente, sum(dr.importo) AS fatturatoTotale
    ORDER BY fatturatoTotale DESC
    LIMIT 1
    ```    
    
* **Domanda**:Quali ditte vendono prodotti della famiglia 'CENTRALI FRIGO'?    
* **Query**:
    ```cypher
    MATCH (d:Ditta)<-[:APPARTIENE_A]-(:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo)-[:APPARTIENE_A]->(:Sottofamiglia)-[:INCLUSA_IN]->(fam:Famiglia) 
    WHERE toLower(trim(fam.nome)) = 'centrali frigo' 
    RETURN DISTINCT d.dittaId AS ditta
    ORDER BY ditta
    ```
    
* **Domanda**: "Chi è il cliente che ha acquistato di più in totale?"
* **Query**:
    ```cypher
    MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)
    WHERE dr.importo > 0
    RETURN c.name AS cliente, sum(dr.importo) AS valoreTotale
    ORDER BY valoreTotale DESC
    LIMIT 1
    ``` 

* **Domanda**: "Qual è l'importo totale acquistato dal fornitore 'FORNITORE ESEMPIO SPA' nel 2023?"
* **Query**:
    ```cypher
    MATCH (gf:GruppoFornitore)<-[:RAGGRUPPATO_SOTTO]-(:Fornitore)-[:HA_EMESSO]->(doc:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)
    WHERE toLower(trim(gf.ragioneSociale)) = 'fornitore esempio spa' AND doc.dataEmissione.year = 2023
    RETURN sum(CASE WHEN dr.importo = -9999 THEN 0 ELSE dr.importo END) AS totaleAcquistato
    ```
    
* **Domanda**: "Qual è il prodotto più venduto in generale (per valore)?"
* **Query**:
    ```cypher
    MATCH (:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) 
    WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 
    RETURN a.descrizione AS prodotto, sum(dr.importo) AS valoreVenduto 
    ORDER BY valoreVenduto DESC 
    LIMIT 1
    ```
* **Domanda**: "Qual è l'importo medio per riga documento nelle fatture di vendita?"
    * **Query**:
    ```cypher
    "MATCH (:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) 
    WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 
    RETURN avg(dr.importo) AS importoMedioRiga"    
    ```
    
// --- NUOVO ESEMPIO AVANZATO: Query "Negative" (chi NON ha fatto qualcosa) ---
* **Domanda**: "Elenca i clienti che non hanno mai acquistato l'articolo 'PRODOTTO ESEMPIO'"
* **Query**:
  ```cypher
  MATCH (a:Articolo) WHERE toLower(trim(a.descrizione)) = 'prodotto esempio'
  WITH a
  MATCH (c:Cliente)
  OPTIONAL MATCH (c)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a)
  WHERE dr IS NULL
  RETURN c.name AS cliente
  ORDER BY cliente
  ```    

// --- ESEMPI DI FILTRAGGIO E LISTE ---

* **Domanda**: "Elenca i clienti della ditta '3' che sono in Toscana"
* **Query**:
    ```cypher
    MATCH (d:Ditta {{dittaId: '3'}})<-[:APPARTIENE_A]-(c:Cliente)-[:HAS_ADDRESS]->(l:Luogo)
    WHERE toLower(trim(l.localita)) = 'PERUGIA'
    RETURN DISTINCT c.name AS cliente
    ORDER BY cliente
    ```

* **Domanda**: "Mostrami i 5 prodotti più popolari in termini di quantità venduta"
* **Query**:
  ```cypher
  MATCH (:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo)
  RETURN a.descrizione AS prodotto, sum(dr.quantita) AS quantitaTotale
  ORDER BY quantitaTotale DESC
  LIMIT 5
  ```    

* **Domanda**: "Quali fornitori si trovano nella stessa localita dei nostri 3 clienti con più fatturato?""    
* **Query**:
    ```cypher
    MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)
    WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0
    WITH c, sum(dr.importo) AS fatturatoTotale
    ORDER BY fatturatoTotale DESC
    LIMIT 3
    WITH collect(c) AS clientiTop
    MATCH (cliente)-[:HAS_ADDRESS]->(l:Luogo)
    WHERE cliente IN clientiTop
    WITH collect(DISTINCT l.localita) AS localitaTop
    MATCH (f:Fornitore)-[:SI_TROVA_A]->(l:Luogo)
    WHERE l.localita IN localitaTop
    MATCH (f)-[:RAGGRUPPATO_SOTTO]->(gf:GruppoFornitore)
    RETURN DISTINCT gf.ragioneSociale as fornitore, l.localita as localita
    ORDER BY fornitore
    ```

* **Domanda**: "Mostrami gli articoli contenuti nelle fatture (codice FVC) ricevute dal cliente Ferrini della ditta 1."
* **Query**:
    ```cypher
    MATCH (d:Ditta {{dittaId: '1'}})<-[:APPARTIENE_A]-(c:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo),
          (doc)-[:IS_TYPE]->(dt:DocType {{codice: 'FVC'}})
    WHERE toLower(trim(c.name)) = 'ferrini'
    RETURN DISTINCT a.descrizione AS articolo
    ```
* **Domanda**: "Quale tipologia di documento è presente maggiormente nel sistema?"
* **Query**:
    ```cypher
    MATCH (doc:Documento)
    RETURN doc.tipoOriginale AS tipoDocumento, count(doc) AS conteggio
    ORDER BY conteggio DESC
    LIMIT 1
    ```

* **Domanda**: "Qual è il prodotto più venduto al cliente con più fatturato?"
* **Query**:
    ```cypher
    MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) 
    WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 
    WITH c, sum(dr.importo) AS fatturatoTotale 
    ORDER BY fatturatoTotale DESC LIMIT 1 WITH c 
    MATCH (c)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo)
    RETURN a.descrizione as prodotto, sum(dr.quantita) as quantitaTotale
    ORDER BY quantitaTotale DESC
    LIMIT 1
    ```
// --- ESEMPI SULLA GERARCHIA PRODOTTI ---

* **Domanda**: "Qual è il fatturato della famiglia 'Elettrodomestici'?"
* **Query**:
    ```cypher
    MATCH (fam:Famiglia)<-[:INCLUSA_IN]-(:Sottofamiglia)<-[:APPARTIENE_A]-(:Articolo)<-[:RIGUARDA_ARTICOLO]-(dr:RigaDocumento)
    WHERE toLower(trim(fam.nome)) = 'elettrodomestici' AND dr.tipoValore = 'Fatturato' AND dr.importo > 0
    RETURN sum(dr.importo) AS fatturatoFamiglia
    ```

* **Domanda**: "Qual è la famiglia di prodotti più acquistata dai fornitori?"
* **Query**:
    ```cypher
    MATCH (:Fornitore)-[:HA_EMESSO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(:Articolo)-[:APPARTIENE_A]->(:Sottofamiglia)-[:INCLUSA_IN]->(fam:Famiglia)
    RETURN fam.nome AS famiglia, sum(dr.quantita) AS quantitaTotale
    ORDER BY quantitaTotale DESC
    LIMIT 1    
    ```
* **Domanda**: "Per la famiglia 'Cucine', mostrami il fatturato per ogni sottofamiglia."
* **Query**:
    ```cypher
    MATCH (fam:Famiglia)<-[:INCLUSA_IN]-(sfam:Sottofamiglia)<-[:APPARTIENE_A]-(:Articolo)<-[:RIGUARDA_ARTICOLO]-(dr:RigaDocumento)
    WHERE toLower(trim(fam.nome)) = 'cucine' AND dr.tipoValore = 'Fatturato' AND dr.importo > 0
    RETURN sfam.nome AS sottofamiglia, sum(dr.importo) AS fatturato
    ORDER BY fatturato DESC
    ```
    
 * **Domanda**: "A quanto ho venduto il prodotto 'PRODOTTO ESEMPIO'?"
* **Query**:
    ```cypher
    MATCH (:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) 
    WHERE toLower(trim(a.descrizione)) = 'prodotto esempio' AND dr.tipoValore = 'Fatturato' AND dr.importo > 0 
    RETURN sum(dr.importo) AS totaleVenduto    
    ```
// --- ESEMPI DI ANALISI COMPLESSA (CON WITH) ---

* **Domanda**: "Chi è il fornitore più importante per ogni ditta?"
* **Query**:
    ```cypher
    MATCH (d:Ditta)<-[:APPARTIENE_A]-(f:Fornitore)-[:HA_EMESSO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)
    WHERE dr.importo > 0
    WITH d, f, sum(dr.importo) AS importoTotale
    ORDER BY d.dittaId, importoTotale DESC
    WITH d, collect({{fornitore: f, importo: importoTotale}}) AS fornitoriOrdinati
    MATCH (fornitoriOrdinati[0].fornitore)-[:RAGGRUPPATO_SOTTO]->(gf:GruppoFornitore)
    RETURN d.dittaId AS ditta, gf.ragioneSociale AS fornitoreTop, fornitoriOrdinati[0].importo AS importoTotale
    ```

* **Domanda Analitica Complessa**: "Per la ditta '1', qual è il fornitore da cui abbiamo acquistato di più gli articoli venduti ai nostri 5 clienti top per fatturato?"
* **Query**:
    ```cypher
    MATCH (d:Ditta {{dittaId: '1'}})<-[:APPARTIENE_A]-(c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)
    WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0
    WITH c, sum(dr.importo) AS fatturatoCliente
    ORDER BY fatturatoCliente DESC
    LIMIT 5
    WITH collect(c) AS clientiTop
    MATCH (c)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo)
    WHERE c IN clientiTop
    WITH collect(DISTINCT a) AS articoliRilevanti
    MATCH (gf:GruppoFornitore)<-[:RAGGRUPPATO_SOTTO]-(:Fornitore)-[:HA_EMESSO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo)
    WHERE a IN articoliRilevanti
    RETURN gf.ragioneSociale AS fornitorePrincipale, sum(dr.importo) AS importoAcquistato
    ORDER BY importoAcquistato DESC
    LIMIT 1
    ```

* **Domanda**: "Il fatturato di ogni ditta, nel loro anno migliore"
* **Query**:
    ```cypher
    MATCH (d:Ditta)<-[:APPARTIENE_A]-(:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)
    WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0
    WITH d, doc.dataEmissione.year AS anno, sum(dr.importo) AS fatturatoAnnuale
    ORDER BY d.dittaId, fatturatoAnnuale DESC
    WITH d, collect({{anno: anno, fatturato: fatturatoAnnuale}}) AS fatturatiOrdinati
    RETURN d.dittaId AS ditta, fatturatiOrdinati[0].anno AS annoTop, fatturatiOrdinati[0].fatturato AS fatturatoMassimo
    ```

* **Domanda**: "Attualmente la ditta 2 è in perdita o in crescita?"
* **Query**:
    ```cypher
     MATCH (d:Ditta {{dittaId: '2'}})<-[:APPARTIENE_A]-(:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) 
     WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 AND doc.dataEmissione.year IN [date().year, date().year - 1] 
     RETURN doc.dataEmissione.year AS anno, sum(dr.importo) AS fatturatoAnnuale 
     ORDER BY anno DESC
    ```

---
**Domanda Utente:**
{question}

**Query Cypher:**
"""

QA_GENERATION_TEMPLATE = """Sei un assistente aziendale e analista di dati per Blues System. Il tuo obiettivo è interpretare i dati estratti dal database e presentarli in modo professionale, chiaro e proattivo.

**ISTRUZIONI E REGOLE DI RISPOSTA:**

1.  **STRUTTURA DELLA RISPOSTA**: Ogni risposta deve avere:
    * Una **frase introduttiva** chiara che contestualizzi il risultato rispetto alla domanda.
    * Il **corpo della risposta** con i dati, formattati secondo le regole seguenti.
    * (Opzionale ma consigliato) Una riga finale con **suggerimenti proattivi** per approfondire.

2.  **GESTIONE DEI DATI**:
    * **Dati Vuoti `[]`**: Se i dati sono vuoti, rispondi solo ed esattamente: "Non ho trovato dati corrispondenti alla tua richiesta nel database."
    * **Risposta Plurale a Domanda Singolare**: Se la domanda è al singolare (es. "qual è il cliente...") ma i dati contengono una lista di più risultati, segnalalo nella frase introduttiva (es. "Ho trovato più clienti che corrispondono:").
    * **Sintesi di Liste Lunghe**: Se l'utente non chiede un numero specifico e la lista di risultati ha più di 7 elementi, elenca solo i primi 5 e menziona il totale (es. "Ho trovato 25 risultati. Ecco i primi 5:"). Se l'utente chiede un numero esatto (es. "dammi 10 clienti"), restituisci quel numero.

3.  **FORMATTAZIONE DEL CONTENUTO**:
    * **Elenchi**: Usa sempre una lista puntata in formato Markdown (`-`), con un elemento per riga.
    * **Valori Monetari**: Se un numero rappresenta un valore economico (chiavi come "importo", "fatturato", "costo"), formattalo con due decimali, il separatore per le migliaia e il simbolo "€" (es. `1.234,56 €`).
    * **Quantità e Conteggi**: Se un numero è una quantità o un conteggio, formattalo come numero intero con il separatore per le migliaia, ma **senza il simbolo "€"** (es. `1.234`).

**ESEMPI DI RISPOSTA CORRETTA:**

**Esempio 1 - Risultato Singolo (Multi-Colonna):**
Domanda: "Qual è il prodotto più acquistato dai clienti?"
Dati: `[{{"prodotto": "VITE AUTOFILETTANTE", "totale_quantita": 5500}}]`
Risposta:
Il prodotto più acquistato dai clienti è la **VITE AUTOFILETTANTE**, con un totale di **5.500** unità vendute.
*Potresti voler sapere qual è il prodotto di maggior valore o chi acquista questo articolo.*

**Esempio 2 - Lista Semplice:**
Domanda: "Elenca i clienti della ditta 3 in Toscana"
Dati: `[{{"cliente": "ACME SRL"}}, {{"cliente": "ROSSI SPA"}}]`
Risposta:
Ho trovato 2 clienti per la ditta 3 in Toscana:
- ACME SRL
- ROSSI SPA
*Vuoi calcolare il loro fatturato totale?*

**Esempio 3 - Lista di Analisi "Drill-Down":**
Domanda: "Per la famiglia 'Cucine', mostrami il fatturato per ogni sottofamiglia."
Dati: `[{{"sottofamiglia": "Forni", "fatturato": 120000.75}}, {{"sottofamiglia": "Piani Cottura", "fatturato": 95000.50}}]`
Risposta:
Ecco il dettaglio del fatturato per le sottofamiglie della famiglia 'Cucine':
- **Forni**: 120.000,75 €
- **Piani Cottura**: 95.000,50 €

**Esempio 4 - Risultato Complesso Multi-Riga:**
Domanda: "Chi è il fornitore più importante per ogni ditta?"
Dati: `[{{"ditta": "1", "fornitoreTop": "FORNITORE A SRL", "importoTotale": 50000}}, {{"ditta": "2", "fornitoreTop": "FORNITORE B SPA", "importoTotale": 75000}}]`
Risposta:
Ecco i fornitori più importanti per ogni ditta, calcolati sull'importo totale acquistato:
- **Ditta 1**: Il fornitore top è **FORNITORE A SRL** con un totale di 50.000,00 €.
- **Ditta 2**: Il fornitore top è **FORNITORE B SPA** con un totale di 75.000,00 €.

---
**DATI ESTRATTI DAL DATABASE:**
{context}

**DOMANDA DELL'UTENTE:**
{question}

**RISPOSTA PROFESSIONALE:**"""


def extract_cypher(text: str) -> str:
    """
    Estrae la query Cypher da una stringa, anche se è avvolta in blocchi di codice Markdown.
    """
    # Cerca un blocco di codice Cypher
    match = re.search(r"```(?:cypher)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Se non trova un blocco, assume che la risposta sia già la query pulita
    return text.strip()


## --- 1. CONFIGURAZIONE INIZIALE ---
cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)
qa_prompt = PromptTemplate(
    input_variables=["context", "question"], template=QA_GENERATION_TEMPLATE
)


def formatta_contesto_per_llm(context: list) -> str:
    """
    Prende il risultato grezzo da Neo4j (lista di dizionari) e lo trasforma
    in una tabella Markdown leggibile per l'LLM. Funziona con qualsiasi dato.
    """
    if not context:
        return "Nessun dato trovato."

    # Estrai gli header dalle chiavi del primo dizionario
    headers = list(context[0].keys())

    # Crea la riga degli header
    header_row = "| " + " | ".join(headers) + " |"
    # Crea la riga di separazione
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

    # Crea le righe di dati
    data_rows = []
    for row in context:
        data_row = "| " + " | ".join([str(row.get(h, "")) for h in headers]) + " |"
        data_rows.append(data_row)

    return "\n".join([header_row, separator_row] + data_rows)


# --- 3. CREAZIONE DELLA CATENA LANGCHAIN ---

cypher_qa_chain = GraphCypherQAChain.from_llm(
    cypher_llm=llm,
    qa_llm=llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt,
    qa_prompt=qa_prompt,
    allow_dangerous_requests=True,
    cypher_query_cleaner=extract_cypher,
)

# --- 4. ENDPOINT API CON CACHE ---


class UserQuestionWithSession(BaseModel):
    question: str
    user_id: Optional[str] = (
        "default_user"  # Identificativo utente opzionale per la sessione
    )


# =======================================================================
@app.get("/health")
async def health_check():
    """Endpoint per verificare lo stato dell'API"""
    try:
        # Ottieni lo stato del database Neo4j
        graph.query("RETURN 1")
        return {
            "status": "online",
            "message": "API BlueAI operativa",
            "neo4j": "connected",
        }
    except Exception as e:
        logger.error(f"Health check fallito: {e}")
        return {"status": "error", "message": str(e), "neo4j": "disconnected"}


DISAMBIGUATION_PROMPT = """
Task: Sei un analista logico. Il tuo compito è analizzare la domanda di un utente e scegliere l'entità più probabile tra i candidati forniti, basandoti sul contesto.

Domanda Utente: "{question}"

Candidati Trovati per il nome "{name}":
{candidates_list}
# Esempio di formato per candidates_list:
# - Candidato 1: Tipo=Cliente, Ditta=1
# - Candidato 2: Tipo=Fornitore

Analizzando parole chiave nella domanda come "fatturato", "venduto", "cliente", "costi", "acquistato", "fornitore", quale candidato è quello corretto?
Rispondi SOLO con il numero del candidato (es. "1"). Se è impossibile decidere con certezza, rispondi "incerto".
"""

MAX_CONTEXT_FOR_LLM = 10


@app.post("/ask")
async def ask_question(user_question: UserQuestionWithSession):
    start_total_time = time.perf_counter()
    timing_details = {}

    user_id = user_question.user_id or "default_user"
    original_question_text = user_question.question.strip()
    question_text = original_question_text
    session = get_or_create_session(user_id)

    logger.info(f"📥 [{user_id}] Domanda: '{question_text}'")

    cache_key = (user_id, question_text)
    if cache_key in query_cache:
        logger.info(f"✅ Cache HIT! Restituisco la risposta salvata.")
        return query_cache[cache_key]

    logger.info(f" Cache MISS. Avvio il processo di analisi.")

    # STEP 1: Intent Classification
    intent = classify_intent(question_text)
    logger.info(f" Intent: {intent}")

    if intent == "CONVERSATIONAL":
        risposta = "Ciao! Sono qui per aiutarti ad analizzare i dati aziendali. Puoi chiedermi informazioni su clienti, fornitori, fatturati e altro."
        return {"domanda": question_text, "risposta": risposta, "intent": intent}

    elif intent == "SYSTEM_HELP":
        risposta = "Posso aiutarti a:\n- Analizzare fatturati e costi\n- Cercare clienti e fornitori\n- Esplorare documenti e articoli\n\nProva: 'Qual è il fatturato di [cliente]?'"
        return {"domanda": question_text, "risposta": risposta, "intent": intent}

    elif intent == "CLARIFICATION":
        if not session.messages:
            return {
                "domanda": question_text,
                "risposta": "Non ho una conversazione precedente. Puoi riformulare?",
            }
        last_answer = session.messages[-1].answer
        risposta = f"Riguardo a: {last_answer}\n\nCosa vuoi sapere esattamente?"
        return {"domanda": question_text, "risposta": risposta, "intent": intent}

    # STEP 2: Fuzzy Matching
    question_corretta, correzioni = correggi_nomi_fuzzy_neo4j(question_text)
    if correzioni:
        logger.info(f" Correzioni: {', '.join(correzioni)}")
        question_text = question_corretta

    # STEP 3: Gestione Chiarimento Disambiguazione
    is_clarification_response = False
    if session.messages:
        last_message = session.messages[-1]
        if last_message.answer.startswith("Ho trovato più voci per"):
            is_clarification_response = True
            original_question = last_message.question
            refine_prompt = f"Combina '{original_question}' con '{question_text}' in una domanda unica."
            refined_question = llm.invoke(refine_prompt).content.strip()
            question_text = refined_question
            logger.info(f"🔄 Riformulata: '{question_text}'")

    # STEP 4: Generazione Cypher + Esecuzione + Formattazione Risposta
    if not is_clarification_response:
        # Estrazione nomi rimane uguale...
        extract_prompt = "..."  # (Prompt di estrazione nomi omesso per brevità)
        extracted_names_str = llm.invoke(extract_prompt).content.strip()

        if extracted_names_str and extracted_names_str.upper() != "NESSUNO":
            for name in extracted_names_str.split(","):
                clean_name = name.strip()
                if len(clean_name) < 3:
                    continue

                # Query per trovare candidati (ora usa i parametri!)
                disambiguation_query = """
                MATCH (n) WHERE toLower(trim(n.name)) = toLower(trim($name)) OR toLower(trim(n.ragioneSociale)) = toLower(trim($name))
                WITH n, labels(n) as tipi
                OPTIONAL MATCH (n)-[:APPARTIENE_A]->(d:Ditta)
                RETURN n.name as official_name, tipi, d.dittaId AS dittaId
                """
                candidates_raw = graph.query(disambiguation_query, {"name": clean_name})

                # Raggruppa per evitare duplicati
                unique_entities = list(
                    {frozenset(item.items()): item for item in candidates_raw}.values()
                )

                if len(unique_entities) > 1:
                    logger.warning(
                        f" Ambiguità rilevata per '{clean_name}'. Avvio risoluzione automatica."
                    )

                    # 1. Formatta i candidati per l'AI-giudice
                    options_for_prompt = []
                    for i, entity in enumerate(unique_entities, 1):
                        main_label = "Entità"
                        if "Cliente" in entity["tipi"]:
                            main_label = "Cliente"
                        elif "GruppoFornitore" in entity["tipi"]:
                            main_label = "Fornitore"
                        ditta_info = (
                            f"della ditta {entity['dittaId']}"
                            if entity["dittaId"]
                            else "non associato a ditta"
                        )
                        options_for_prompt.append(
                            f"- Candidato {i}: Tipo={main_label}, {ditta_info}"
                        )

                    candidates_list_str = "\n".join(options_for_prompt)

                    # 2. Chiedi all'AI-giudice di scegliere
                    prompt_for_judge = DISAMBIGUATION_PROMPT.format(
                        question=question_text,
                        name=clean_name,
                        candidates_list=candidates_list_str,
                    )

                    ai_choice_str = llm.invoke(prompt_for_judge).content.strip()

                    chosen_entity = None
                    if ai_choice_str.isdigit():
                        choice_index = int(ai_choice_str) - 1
                        if 0 <= choice_index < len(unique_entities):
                            chosen_entity = unique_entities[choice_index]
                            logger.info(
                                f" AI-giudice ha scelto il candidato #{ai_choice_str}: {chosen_entity}"
                            )

                    # 3. Riformulazione della domanda o chiarimenti
                    if chosen_entity:
                        # L'AI è sicura della scelta, riformulazione della domanda
                        chosen_name = chosen_entity["official_name"]
                        chosen_type = (
                            "Cliente"
                            if "Cliente" in chosen_entity["tipi"]
                            else "Fornitore"
                        )
                        ditta_id = chosen_entity.get("dittaId")

                        specifier = f"il {chosen_type} '{chosen_name}'"
                        if ditta_id:
                            specifier += f" della ditta {ditta_id}"

                        refine_prompt = f"Riscrivi la domanda seguente, sostituendo il riferimento a '{clean_name}' con la specifica: '{specifier}'. Domanda originale: '{question_text}'"
                        question_text = llm.invoke(refine_prompt).content.strip()
                        logger.info(f" Domanda disambiguata dall'AI: '{question_text}'")
                    else:
                        # Fallback: l'AI non è sicura, quindi chiediamo all'utente come prima
                        logger.warning(f" Ambiguità: '{name.strip()}'")
                    # --- NUOVA LOGICA DI COSTRUZIONE RISPOSTA ---
                    intro_response = f"Ho trovato più corrispondenze per '{name.strip()}'. Per favore, specifica a quale ti riferisci:"
                    options_list = []
                    for i, entity in enumerate(unique_entities.values(), 1):
                        main_label = "l'Entità"
                        if "Cliente" in entity["tipi"]:
                            main_label = "il Cliente"
                        elif "GruppoFornitore" in entity["tipi"]:
                            main_label = "il Fornitore"

                        ditta_info = (
                            f"della ditta {entity['dittaId']}"
                            if entity["dittaId"]
                            else "non associato a una ditta"
                        )
                        options_list.append(f"{i}. {main_label} {ditta_info}")

                    options_text = "\n".join(options_list)
                    disambiguation_response = f"{intro_response}\n{options_text}\n\nPuoi rispondere semplicemente con il numero dell'opzione."
                    add_message_to_session(
                        user_id,
                        user_question.question.strip(),  # Domanda originale
                        disambiguation_response,
                        [],
                        "Disambiguazione richiesta",
                    )

                    return {
                        "domanda": user_question.question.strip(),
                        "risposta": disambiguation_response,
                        "needs_clarification": True,
                    }
    # STEP 5: Esecuzione RAG
    try:
        # 5.1: Prepara il payload per il prompt unificato. È SEMPRE lo stesso.
        conversation_context = get_conversation_context(user_id)

        # Il prompt template è sempre UNIFIED_CYPHER_TEMPLATE
        unified_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

        input_payload = {
            "question": question_text,
            "conversation_context": conversation_context,
        }
        logger.info(
            f"🧠 Contesto conversazione inviato al prompt: '{conversation_context}'"
        )

        # 5.2: Generazione della query Cypher (ora la logica è lineare e semplice)
        start_gen_time = time.perf_counter()
        cypher_chain = unified_prompt | llm | StrOutputParser() | extract_cypher
        generated_query = cypher_chain.invoke(input_payload)
        timing_details["generazione_query"] = time.perf_counter() - start_gen_time
        logger.info(
            f"🤖 Query Cypher Generata in {timing_details['generazione_query']:.2f}s:\n{generated_query}"
        )

        # 5.3: Esecuzione della query (logica invariata)
        start_db_time = time.perf_counter()
        context_completo = graph.query(generated_query)
        timing_details["esecuzione_db"] = time.perf_counter() - start_db_time
        logger.info(
            f"📊 Context Trovato dal DB in {timing_details['esecuzione_db']:.2f}s: {len(context_completo)} risultati"
        )

        # 5.4: Sintesi della risposta finale (logica invariata)
        final_answer = ""
        if not context_completo:
            final_answer = (
                "Non ho trovato dati corrispondenti alla tua richiesta nel database."
            )
            timing_details["sintesi_risposta"] = 0.0
        else:
            # Troncamento per LLM di sintesi, se necessario
            context_da_inviare = context_completo[:MAX_CONTEXT_FOR_LLM]
            start_qa_time = time.perf_counter()
            qa_chain = qa_prompt | llm
            final_answer_raw = qa_chain.invoke(
                {"context": context_da_inviare, "question": original_question_text}
            )
            final_answer = final_answer_raw.content.strip()
            timing_details["sintesi_risposta"] = time.perf_counter() - start_qa_time

        # 5.5: Fallback e salvataggio (logica invariata)
        is_answer_valid = final_answer and "non ho trovato" not in final_answer.lower()
        if not is_answer_valid and context_completo:
            logger.warning("⚠️ L'AI ha fallito la sintesi, uso il fallback hardcoded.")
            final_answer = create_fallback_response(
                context_da_inviare, original_question_text
            )

        logger.info(f"✅ Risposta Finale: {final_answer}")

        # Aggiungi il messaggio alla sessione e alla cache
        add_message_to_session(
            user_id,
            original_question_text,
            final_answer,
            context_completo,
            generated_query,
        )

        response_payload = {
            "domanda": original_question_text,
            "query_generata": generated_query,
            "risposta": final_answer,
            "context_trovato": context_completo[:5],
            "intent": intent,
            "timing_details": timing_details,
        }
        query_cache[cache_key] = response_payload
        return response_payload

    except Exception as e:
        logger.error(
            f"❌ ERRORE GRAVE nel flusso RAG per {user_id}: {e}", exc_info=True
        )
        error_msg = "Si è verificato un errore imprevisto."
        if "SyntaxError" in str(e):
            error_msg = "L'AI ha generato una query con un errore di sintassi. Prova a riformulare la domanda."
        return {"domanda": original_question_text, "risposta": error_msg, "error": True}


# Nuovo endpoint per visualizzare la conversazione
@app.get("/conversation/{user_id}")
async def get_conversation(user_id: str):
    """Ottieni la cronologia della conversazione per un utente"""
    session = conversation_sessions.get(user_id)
    if not session:
        return {"messages": [], "total": 0}

    return {
        "user_id": user_id,
        "messages": [
            {
                "timestamp": msg.timestamp.isoformat(),
                "question": msg.question,
                "answer": msg.answer,
                "had_context": len(msg.context) > 0,
            }
            for msg in session.messages
        ],
        "total": len(session.messages),
        "session_created": session.created_at.isoformat(),
        "last_activity": session.last_activity.isoformat(),
    }


# Endpoint per cancellare la memoria di un utente
@app.delete("/conversation/{user_id}")
async def clear_conversation(user_id: str):
    """Cancella la cronologia conversazione di un utente"""
    if user_id in conversation_sessions:
        del conversation_sessions[user_id]
        return {"message": f"Conversazione cancellata per {user_id}"}
    return {"message": f"Nessuna conversazione trovata per {user_id}"}


@app.delete("/cache")
async def clear_cache():
    """
    Endpoint per svuotare TUTTE le cache: risposte e memoria conversazioni.
    """
    global query_cache
    global conversation_sessions

    logger.info(" Richiesta di svuotamento manuale di TUTTE le cache...")
    try:
        # Svuota la cache delle risposte
        responses_removed = query_cache.currsize
        query_cache.clear()

        # Svuota la cache delle sessioni di conversazione
        sessions_removed = len(conversation_sessions)
        conversation_sessions.clear()

        logger.info(
            f"✅ Cache svuotata con successo. (Risposte rimosse: {responses_removed}, Sessioni rimosse: {sessions_removed})"
        )
        return {
            "message": "Tutte le cache (risposte e conversazioni) sono state svuotate.",
            "risposte_rimosse": responses_removed,
            "sessioni_rimosse": sessions_removed,
        }
    except Exception as e:
        logger.error(f" Errore durante lo svuotamento della cache: {e}")
        return {
            "message": "Errore durante lo svuotamento della cache.",
            "error": str(e),
        }
