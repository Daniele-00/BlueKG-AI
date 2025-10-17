import pyodbc
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from decimal import Decimal
from datetime import datetime, date
import time
import re

# --- CONFIGURAZIONE DEI DATABASE ---
# Inserisci qui le credenziali corrette per MS SQL Server
# SQL_SERVER_GTG-FRANEW = '172.16.200.136'
SQL_SERVER = "172.16.200.150"
# SQL_DATABASE = 'GTG-FRANEW'
SQL_DATABASE = "METALMECCANICA_FALSO(UMBRIAFRIGO)"
SQL_USERNAME = "sa"
SQL_PASSWORD = "sa.12345"
# La "connection string" per connettersi a SQL Server da Linux
SQL_CONN_STR = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={SQL_SERVER};"
    f"DATABASE={SQL_DATABASE};"
    f"UID={SQL_USERNAME};"
    f"PWD={SQL_PASSWORD};"
    f"TrustServerCertificate=yes;"  # Necessario per certificati auto-firmati
)

# Configurazione di Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "Blue2018")

# Valori di default per indicare dati mancanti
DEFAULT_VALUES = {
    "string": "DATO_NON_DISPONIBILE",
    "numeric": 0.0,
    "id": "ID_NON_DISPONIBILE",
    "date": "1900-01-01",  # Data di default per date mancanti
}

# Aggiungi questa mappa nella sezione CONFIGURAZIONE del tuo script

# =======================================================================
# PARTE DA SOSTITUIRE: L'INTERA SEZIONE DELLE MAPPE DOCUMENTI
# Usa questa mappa unificata e definitiva
# =======================================================================
MAPPA_DOCTYPES = {
    # --- Mappatura Documenti Cliente ---
    "FATT. IMMEDIATA VENDITA": {"codice": "FVC", "nome": "Fattura Cliente"},
    "FATT. IMMEDIATA VENDITA PERUGIA": {"codice": "FVC", "nome": "Fattura Cliente"},
    "FATT. IMMEDIATA VENDITA ROMA": {"codice": "FVC", "nome": "Fattura Cliente"},
    "FATT. IMMEDIATA VENDITA PROVA MAG": {"codice": "FVC", "nome": "Fattura Cliente"},
    "FATT. IMMEDIATA VENDITA TD26": {"codice": "FVC", "nome": "Fattura Cliente"},
    "FATT. RIEPILOGATIVA CLIENTE": {"codice": "FVC", "nome": "Fattura Cliente"},
    "FATT. RIEPILOGATIVA CLIENTE PERUGIA": {"codice": "FVC", "nome": "Fattura Cliente"},
    "FATT. RIEPILOGATIVA CLIENTE ROMA": {"codice": "FVC", "nome": "Fattura Cliente"},
    "FATTURA D'ACCONTO PERUGIA": {"codice": "FVC", "nome": "Fattura Cliente"},
    "FATTURA D'ACCONTO ROMA": {"codice": "FVC", "nome": "Fattura Cliente"},
    "Fattura vendita con ritenute UF": {"codice": "FVC", "nome": "Fattura Cliente"},
    "PREFATTURAZION CONTRATTI PERUGIA": {"codice": "FVC", "nome": "Fattura Cliente"},
    "PREFATTURAZION CONTRATTI ROMA": {"codice": "FVC", "nome": "Fattura Cliente"},
    "FATT. AUTOCONSUMO PERUGIA": {"codice": "FVC", "nome": "Fattura Cliente"},
    "DDT A CLIENTE NF": {"codice": "DDTV", "nome": "DdT Cliente"},
    "DDT C/VENDITA": {"codice": "DDTV", "nome": "DdT Cliente"},
    "DDT C/VENDITA PERUGIA": {"codice": "DDTV", "nome": "DdT Cliente"},
    "DDT C/VENDITA ROMA": {"codice": "DDTV", "nome": "DdT Cliente"},
    "DDT C/VISIONE A CLIENTE": {"codice": "DDTV", "nome": "DdT Cliente"},
    "DDT INSTALLAZIONE": {"codice": "DDTV", "nome": "DdT Cliente"},
    "DDT INSTALLAZIONE PERUGIA": {"codice": "DDTV", "nome": "DdT Cliente"},
    "DDT INSTALLAZIONE ROMA": {"codice": "DDTV", "nome": "DdT Cliente"},
    "ORDINE DA CLIENTE - PERUGIA": {"codice": "OC", "nome": "Ordine Cliente"},
    "ORDINE DA CLIENTE - ROMA": {"codice": "OC", "nome": "Ordine Cliente"},
    "NOTA DI CREDITO VENDITA": {"codice": "NCC", "nome": "Nota Credito Cliente"},
    "NOTA DI CREDITO VENDITA PERUGIA": {
        "codice": "NCC",
        "nome": "Nota Credito Cliente",
    },
    "NOTA DI CREDITO VENDITA ROMA": {"codice": "NCC", "nome": "Nota Credito Cliente"},
    "NOTA DI CREDITO S/MAGAZZINO": {"codice": "NCC", "nome": "Nota Credito Cliente"},
    "NOTA DI CREDITO S/MAGAZZINO PERUGIA": {
        "codice": "NCC",
        "nome": "Nota Credito Cliente",
    },
    "NOTA DI CREDITO S/MAGAZZINO ROMA": {
        "codice": "NCC",
        "nome": "Nota Credito Cliente",
    },
    "NOTA DEBITO VENDITA PERUGIA": {"codice": "NDC", "nome": "Nota Debito Cliente"},
    "NOTA DEBITO VENDITA ROMA": {"codice": "NDC", "nome": "Nota Debito Cliente"},
    "PREVENTIVO CLIENTE - PERUGIA": {"codice": "PREV-C", "nome": "Preventivo Cliente"},
    "PREVENTIVO CLIENTE - ROMA": {"codice": "PREV-C", "nome": "Preventivo Cliente"},
    # --- Mappatura Documenti Fornitore ---
    "Fattura immediata acquisto no coge": {
        "codice": "FFA",
        "nome": "Fattura Fornitore",
    },
    "Fattura immediata acquisto sp": {"codice": "FFA", "nome": "Fattura Fornitore"},
    "Fattura (autofattura) TD16 senza Coge": {
        "codice": "FFA",
        "nome": "Fattura Fornitore",
    },
    "Fattura (autofattura) TD17 senza Coge": {
        "codice": "FFA",
        "nome": "Fattura Fornitore",
    },
    "Fattura (autofattura) TD18 senza Coge": {
        "codice": "FFA",
        "nome": "Fattura Fornitore",
    },
    "DDT A FORNITORE NF": {"codice": "DDTA", "nome": "DdT Fornitore"},
    "DDT DI ACQUISTO PERUGIA": {"codice": "DDTA", "nome": "DdT Fornitore"},
    "DDT DI ACQUISTO ROMA": {"codice": "DDTA", "nome": "DdT Fornitore"},
    "DDT RESO A FORNITORE": {"codice": "DDTA", "nome": "DdT Fornitore"},
    "DDT RESO A FORNITORE PERUGIA": {"codice": "DDTA", "nome": "DdT Fornitore"},
    "DDT RESO A FORNITORE ROMA": {"codice": "DDTA", "nome": "DdT Fornitore"},
    "ORDINE A FORNITORE": {"codice": "OF", "nome": "Ordine Fornitore"},
    "ORDINE A FORNITORE PERUGIA": {"codice": "OF", "nome": "Ordine Fornitore"},
    "ORDINE A FORNITORE ROMA": {"codice": "OF", "nome": "Ordine Fornitore"},
    "RICHIESTA PREV. FORNITORE PER ACQUISTO": {
        "codice": "PREV-F",
        "nome": "Richiesta Preventivo Fornitore",
    },
    # Aggiungi qui altre mappature che non ho intercettato
}
DEFAULT_DOCTYPE = {"codice": "ALTRO", "nome": "Altro Documento"}
# Dimensione batch per il caricamento
BATCH_SIZE = 5000


def converti_tipi_per_neo4j(data):
    """
    Converte ricorsivamente tutti i tipi non supportati da Neo4j
    """
    if isinstance(data, dict):
        return {k: converti_tipi_per_neo4j(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [converti_tipi_per_neo4j(item) for item in data]
    elif isinstance(data, Decimal):
        return float(data)
    elif isinstance(data, datetime):
        return data.isoformat()[:10]  # Converte a stringa YYYY-MM-DD
    elif isinstance(data, date):
        return data.isoformat()  # Converte a stringa YYYY-MM-DD
    elif data is None:
        return None
    else:
        return data


def pulisci_record_fornitori(fornitori):
    """
    Pulisce i record dei fornitori, rimuove spazi e gestisce i null.
    """
    print("🧹 Pulizia record fornitori in corso...")
    fornitori_puliti = []

    for forn in fornitori:
        forn_pulito = forn.copy()

        # Saltiamo record senza ID
        if not forn_pulito.get("_keyfornitore"):
            continue

        def clean_str(value):
            if isinstance(value, str):
                return re.sub(r"\s+", " ", value).strip()
            return value

        # Convertiamo l'ID della ditta in una stringa per consistenza
        if forn_pulito.get("codice_ditta") is not None:
            forn_pulito["codice_ditta"] = str(int(forn_pulito["codice_ditta"]))
        else:
            forn_pulito["codice_ditta"] = DEFAULT_VALUES["id"]

        # Pulizia campi stringa
        forn_pulito["ragione_sociale"] = (
            clean_str(forn_pulito.get("ragione_sociale")) or DEFAULT_VALUES["string"]
        )
        forn_pulito["localita"] = (
            clean_str(forn_pulito.get("localita")) or DEFAULT_VALUES["string"]
        )
        forn_pulito["provincia"] = (
            clean_str(forn_pulito.get("provincia")) or DEFAULT_VALUES["string"]
        )
        forn_pulito["regione"] = (
            clean_str(forn_pulito.get("regione")) or DEFAULT_VALUES["string"]
        )
        forn_pulito["nazione"] = (
            clean_str(forn_pulito.get("nazione")) or DEFAULT_VALUES["string"]
        )

        fornitori_puliti.append(forn_pulito)

    print(f"✅ Pulizia completata: {len(fornitori_puliti)} fornitori validi.")
    return fornitori_puliti


def carica_fornitori_batch(tx, fornitori_batch, batch_num, total_batches):
    print(
        f"      📦 Batch {batch_num}/{total_batches}: caricando {len(fornitori_batch)} anagrafiche fornitore..."
    )

    query = """
    UNWIND $rows AS row

    // --- 1. MERGE sul Gruppo Fornitore (basato sul nome) ---
    // Questo è il nodo "concettuale"
    MERGE (gf:GruppoFornitore {ragioneSociale: row.ragione_sociale})
    ON CREATE SET gf.dataCaricamento = datetime()

    // --- 2. MERGE sul Fornitore specifico (basato sull'ID univoco) ---
    // Questa è l'anagrafica che ha le transazioni
    MERGE (f:Fornitore {fornitoreId: row._keyfornitore})
    ON CREATE SET
        f.ragioneSociale = row.ragione_sociale, // Duplichiamo il nome per comodità
        f.dataCaricamento = datetime()

    // --- 3. MERGE su Ditta e Luogo (invariato) ---
    MERGE (di:Ditta {dittaId: row.codice_ditta})
    MERGE (l:Luogo {localita: row.localita, provincia: row.provincia, regione: row.regione, nazione: row.nazione})

    // --- 4. Crea tutte le relazioni ---
    MERGE (f)-[:RAGGRUPPATO_SOTTO]->(gf)
    MERGE (f)-[:APPARTIENE_A]->(di)
    MERGE (f)-[:SI_TROVA_A]->(l)
    """

    start_time = time.time()
    tx.run(query, rows=fornitori_batch)
    elapsed = time.time() - start_time
    print(f"      ✅ Batch {batch_num} completato in {elapsed:.2f}s")


def carica_fornitori(tx, fornitori):
    """
    Funzione principale che gestisce la pulizia e il caricamento a batch dei fornitori.
    """
    print(f"🚀 Inizio caricamento anagrafica di {len(fornitori)} fornitori...")

    fornitori_puliti = pulisci_record_fornitori(fornitori)
    fornitori_converted = [converti_tipi_per_neo4j(forn) for forn in fornitori_puliti]

    total_batches = (len(fornitori_converted) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"📊 Caricamento in {total_batches} batch da {BATCH_SIZE} record...")

    for i in range(0, len(fornitori_converted), BATCH_SIZE):
        batch = fornitori_converted[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        carica_fornitori_batch(tx, batch, batch_num, total_batches)


def pulisci_record_clienti(clienti):
    """
    Pulisce i record dei clienti, rimuove spazi e sostituisce i valori null.
    """
    print(" Pulizia record clienti in corso...")
    clienti_puliti = []
    record_scartati = 0

    for i, cliente in enumerate(clienti):
        if (i + 1) % 5000 == 0:
            print(f"   Processati {i + 1}/{len(clienti)} clienti...")

        cliente_pulito = cliente.copy()

        # Gestione ID DITTA
        if cliente_pulito.get("codice_ditta") is not None:
            cliente_pulito["codice_ditta"] = str(int(cliente_pulito["codice_ditta"]))
        else:
            cliente_pulito["codice_ditta"] = DEFAULT_VALUES["id"]

        # Campi obbligatori per MERGE
        if not cliente_pulito.get("_keycliente"):
            record_scartati += 1
            continue

        # Funzione helper per pulire le stringhe
        def clean_str(value):
            if isinstance(value, str):
                return re.sub(r"\s+", " ", value).strip()
            return value

        # Gestione campi per il luogo (usati nel MERGE)
        cliente_pulito["localita"] = (
            clean_str(cliente_pulito.get("localita")) or DEFAULT_VALUES["string"]
        )
        cliente_pulito["provincia"] = (
            clean_str(cliente_pulito.get("provincia")) or DEFAULT_VALUES["string"]
        )
        cliente_pulito["nazione"] = (
            clean_str(cliente_pulito.get("nazione")) or DEFAULT_VALUES["string"]
        )
        cliente_pulito["regione"] = (
            clean_str(cliente_pulito.get("regione")) or DEFAULT_VALUES["string"]
        )

        # Altri campi stringa
        cliente_pulito["ragione_sociale"] = (
            clean_str(cliente_pulito.get("ragione_sociale")) or DEFAULT_VALUES["string"]
        )
        cliente_pulito["tipoconto"] = (
            clean_str(cliente_pulito.get("tipoconto")) or DEFAULT_VALUES["string"]
        )
        cliente_pulito["codice_pagamento"] = (
            clean_str(cliente_pulito.get("codice_pagamento"))
            or DEFAULT_VALUES["string"]
        )

        # Campi numerici
        if cliente_pulito.get("fido_euro") is None:
            cliente_pulito["fido_euro"] = DEFAULT_VALUES["numeric"]

        clienti_puliti.append(cliente_pulito)

    print(
        f" Pulizia completata: {len(clienti_puliti)} validi, {record_scartati} scartati"
    )
    return clienti_puliti


# SOSTITUISCI L'INTERA FUNZIONE CON QUESTA
def pulisci_record_documenti(documenti):
    """
    Pulisce i record, rimuove spazi, sostituisce i null, assegna una categoria
    e standardizza i campi di importo in 'importo' e 'tipoValore'.
    """
    print(
        "Pulizia, categorizzazione e standardizzazione record documenti cliente in corso..."
    )
    documenti_puliti = []
    record_scartati = 0

    for doc in documenti:
        doc_pulito = doc.copy()

        if not doc_pulito.get("_keycliente") or not doc_pulito.get("_keyidtidr"):
            record_scartati += 1
            continue

        def clean_str(value):
            if isinstance(value, str):
                return re.sub(r"\s+", " ", value).strip()
            return value

        doc_pulito["_keyarticolo"] = (
            clean_str(doc_pulito.get("_keyarticolo")) or DEFAULT_VALUES["id"]
        )
        doc_pulito["_keydocumento"] = (
            clean_str(doc_pulito.get("_keydocumento")) or DEFAULT_VALUES["id"]
        )
        doc_pulito["tipo_documento"] = (
            clean_str(doc_pulito.get("tipo_documento")) or DEFAULT_VALUES["string"]
        )

        tipo_originale = doc_pulito.get("tipo_documento", "")
        doc_pulito["doctype_mappato"] = MAPPA_DOCTYPES.get(
            tipo_originale, DEFAULT_DOCTYPE
        )

        if doc_pulito.get("data_documento") is None:
            doc_pulito["data_documento"] = DEFAULT_VALUES["date"]

        # --- LOGICA DI STANDARDIZZAZIONE DEL VALORE ---
        # Priorità: Fatturato > Bollato > Ordinato
        # Inizializziamo i nuovi campi
        doc_pulito["importo"] = DEFAULT_VALUES["numeric"]
        doc_pulito["tipoValore"] = "Non Specificato"

        val_fatturato = doc_pulito.get("importo_nettissimo_fatturato")
        val_bolla = doc_pulito.get("importo_nettissimo_bolla")
        val_ordinato = doc_pulito.get("importo_nettissimo_ordinato")

        if val_fatturato is not None and val_fatturato > 0:
            doc_pulito["importo"] = val_fatturato
            doc_pulito["tipoValore"] = "Fatturato"
        elif val_bolla is not None and val_bolla > 0:
            doc_pulito["importo"] = val_bolla
            doc_pulito["tipoValore"] = "Bollato"
        elif val_ordinato is not None and val_ordinato > 0:
            doc_pulito["importo"] = val_ordinato
            doc_pulito["tipoValore"] = "Ordinato"

        # Pulizia della quantità (qta_fatturata è la più rilevante per il cliente)
        doc_pulito["quantita"] = (
            doc_pulito.get("qta_fatturata") or DEFAULT_VALUES["numeric"]
        )

        documenti_puliti.append(doc_pulito)

    print(
        f"✅ Pulizia completata: {len(documenti_puliti)} validi, {record_scartati} scartati"
    )
    return documenti_puliti


# SOSTITUISCI L'INTERA FUNZIONE CON QUESTA
# SOSTITUISCI L'INTERA FUNZIONE CON QUESTA
def pulisci_record_documenti_fornitori(documenti_fornitori):
    """
    Pulisce i record dei documenti fornitori, assegna una categoria,
    standardizza i campi di importo e gestisce ID mancanti.
    """
    print(
        "🧹 Pulizia, categorizzazione e standardizzazione record documenti fornitori in corso..."
    )
    documenti_puliti = []

    for doc in documenti_fornitori:
        doc_pulito = doc.copy()

        # Unica funzione per pulire le stringhe
        def clean_str(value):
            if isinstance(value, str):
                return re.sub(r"\s+", " ", value).strip()
            return value

        # Controllo robusto: saltiamo la riga solo se le chiavi fondamentali mancano
        if not doc_pulito.get("_keyfornitore") or not doc_pulito.get("_keyidtidr"):
            continue

        # --- FIX FONDAMENTALE: Gestione robusta di _keyarticolo ---
        # Se _keyarticolo è nullo o una stringa vuota, usiamo il valore di default.
        doc_pulito["_keyarticolo"] = (
            clean_str(doc_pulito.get("_keyarticolo")) or DEFAULT_VALUES["id"]
        )
        # --- FINE FIX ---

        doc_pulito["tipo_documento"] = (
            clean_str(doc_pulito.get("tipo_documento")) or DEFAULT_VALUES["string"]
        )

        tipo_originale = doc_pulito.get("tipo_documento", "")
        # Assicurati che MAPPA_DOCTYPES e DEFAULT_DOCTYPE siano definiti globalmente
        doc_pulito["doctype_mappato"] = MAPPA_DOCTYPES.get(
            tipo_originale, DEFAULT_DOCTYPE
        )

        # Logica di standardizzazione del valore
        doc_pulito["importo"] = (
            doc_pulito.get("importo_netto_ordinato") or DEFAULT_VALUES["numeric"]
        )
        doc_pulito["tipoValore"] = "Ordinato"
        doc_pulito["quantita"] = (
            doc_pulito.get("qta_ordinata") or DEFAULT_VALUES["numeric"]
        )

        documenti_puliti.append(doc_pulito)

    print(f"✅ Pulizia completata: {len(documenti_puliti)} documenti fornitori validi.")
    return documenti_puliti


# SOSTITUISCI L'INTERA FUNZIONE CON QUESTA
def carica_documenti_fornitori_batch(tx, doc_forn_batch, batch_num, total_batches):
    print(
        f"      📦 Batch {batch_num}/{total_batches}: caricando {len(doc_forn_batch)} righe documento fornitore..."
    )

    query = """
    UNWIND $rows AS row

    // --- 1. "GET OR CREATE" delle entità collegate ---
    MERGE (f:Fornitore {fornitoreId: row._keyfornitore})
      ON CREATE SET f.ragioneSociale = 'Fornitore placeholder: ' + row._keyfornitore, f.placeholder = true

    MERGE (a:Articolo {productId: row._keyarticolo})
      ON CREATE SET a.descrizione = 'Articolo placeholder: ' + row._keyarticolo, a.placeholder = true

    // --- 2. MERGE su DocType ---
    MERGE (dt:DocType {codice: row.doctype_mappato.codice})
    ON CREATE SET dt.nome = row.doctype_mappato.nome

    // --- 3. MERGE sulla Testata del Documento ---
    MERGE (d:Documento {documentoId: row._keydocumento})
      ON CREATE SET d.tipoOriginale = row.tipo_documento, d.dataEmissione = date(row.data_documento)
          
    // --- 4. MERGE sulla Riga del Documento CON I NUOVI CAMPI STANDARD ---
    MERGE (dr:RigaDocumento {rigaId: row._keyidtidr})
      SET
        dr.quantita = toFloat(row.quantita),
        dr.importo = toFloat(row.importo),      // <-- NUOVO CAMPO
        dr.tipoValore = row.tipoValore        // <-- NUOVO CAMPO

    // --- 5. Crea tutte le relazioni ---
    MERGE (d)-[:IS_TYPE]->(dt)
    MERGE (f)-[:HA_EMESSO]->(d)
    MERGE (d)-[:CONTIENE_RIGA]->(dr)
    MERGE (dr)-[:RIGUARDA_ARTICOLO]->(a)
    """

    start_time = time.time()
    tx.run(query, rows=doc_forn_batch)
    elapsed = time.time() - start_time
    print(f"      ✅ Batch {batch_num} completato in {elapsed:.2f}s")


def carica_documenti_fornitori(tx, documenti_fornitori):
    """
    Funzione principale per la pulizia e il caricamento a batch dei documenti fornitore.
    """
    print(f"🚀 Inizio caricamento di {len(documenti_fornitori)} documenti fornitore...")

    doc_forn_puliti = pulisci_record_documenti_fornitori(documenti_fornitori)
    doc_forn_converted = [converti_tipi_per_neo4j(doc) for doc in doc_forn_puliti]

    total_batches = (len(doc_forn_converted) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"📊 Caricamento in {total_batches} batch da {BATCH_SIZE} record...")

    for i in range(0, len(doc_forn_converted), BATCH_SIZE):
        batch = doc_forn_converted[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        carica_documenti_fornitori_batch(tx, batch, batch_num, total_batches)


# SOSTITUISCI L'INTERA FUNZIONE CON QUESTA
def pulisci_record_articoli(articoli):
    """
    Pulisce i record degli articoli, inclusi codici e descrizioni di famiglia e sottofamiglia.
    """
    print("🧹 Pulizia record articoli (con gerarchie) in corso...")
    articoli_puliti = []

    for art in articoli:
        art_pulito = art.copy()

        def clean_str(value):
            if isinstance(value, str):
                return re.sub(r"\s+", " ", value).strip()
            return value

        if not art_pulito.get("_keyarticolo"):
            continue

        art_pulito["descrizione_articolo"] = (
            clean_str(art_pulito.get("descrizione_articolo"))
            or DEFAULT_VALUES["string"]
        )

        # --- NUOVA PARTE: Pulizia Codici e Descrizioni Gerarchia ---
        art_pulito["codice_famiglia"] = (
            clean_str(art_pulito.get("codice_famiglia")) or DEFAULT_VALUES["id"]
        )
        art_pulito["descrizione_famiglia"] = (
            clean_str(art_pulito.get("descrizione_famiglia"))
            or DEFAULT_VALUES["string"]
        )
        art_pulito["codice_sottofamiglia"] = (
            clean_str(art_pulito.get("codice_sottofamiglia")) or DEFAULT_VALUES["id"]
        )
        art_pulito["descrizione_sottofamiglia"] = (
            clean_str(art_pulito.get("descrizione_sottofamiglia"))
            or DEFAULT_VALUES["string"]
        )
        # --- FINE NUOVA PARTE ---

        articoli_puliti.append(art_pulito)

    print(f"✅ Pulizia completata: {len(articoli_puliti)} articoli validi.")
    return articoli_puliti

    print(f"✅ Pulizia completata: {len(articoli_puliti)} articoli validi.")
    return articoli_puliti


# SOSTITUISCI L'INTERA FUNZIONE CON QUESTA
# SOSTITUISCI L'INTERA FUNZIONE CON QUESTA
def carica_articoli_batch(tx, articoli_batch, batch_num, total_batches):
    """
    Carica o aggiorna un batch di articoli, creando la gerarchia basata su codici.
    """
    print(
        f"     📦 Batch {batch_num}/{total_batches}: caricando/aggiornando {len(articoli_batch)} articoli..."
    )

    query = """
    UNWIND $rows AS row

    // Fase 1: MERGE sull'Articolo
    MERGE (a:Articolo {productId: row._keyarticolo})
    SET
        a.descrizione = row.descrizione_articolo,
        a.dataAggiornamento = datetime()

    // Fase 2: Gestione Gerarchia basata su CODICI
    WITH a, row
    // Crea/trova la Sottofamiglia e collega l'Articolo
    WHERE row.codice_sottofamiglia <> $default_id
    MERGE (sf:Sottofamiglia {codice: row.codice_sottofamiglia})
      ON CREATE SET sf.nome = row.descrizione_sottofamiglia
    MERGE (a)-[:APPARTIENE_A]->(sf)

    // Crea/trova la Famiglia e collega la Sottofamiglia
    WITH a, sf, row
    WHERE row.codice_famiglia <> $default_id
    MERGE (f:Famiglia {codice: row.codice_famiglia})
      ON CREATE SET f.nome = row.descrizione_famiglia
    MERGE (sf)-[:INCLUSA_IN]->(f)
    """

    start_time = time.time()
    # Aggiungiamo il parametro $default_id per la query
    tx.run(query, rows=articoli_batch, default_id=DEFAULT_VALUES["id"])
    elapsed = time.time() - start_time
    print(f"     ✅ Batch {batch_num} completato in {elapsed:.2f}s")


def carica_articoli(tx, articoli):
    """
    Funzione principale che gestisce la pulizia e il caricamento a batch degli articoli.
    """
    print(f"🚀 Inizio caricamento anagrafica di {len(articoli)} articoli...")

    articoli_puliti = pulisci_record_articoli(articoli)
    articoli_converted = [converti_tipi_per_neo4j(art) for art in articoli_puliti]

    total_batches = (len(articoli_converted) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"📊 Caricamento in {total_batches} batch da {BATCH_SIZE} record...")

    for i in range(0, len(articoli_converted), BATCH_SIZE):
        batch = articoli_converted[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        carica_articoli_batch(tx, batch, batch_num, total_batches)


def carica_clienti_e_luoghi_batch(tx, clienti_batch, batch_num, total_batches):
    """
    Carica un batch di clienti, mappandoli su un'ontologia ibrida (CDM + Schema.org).
    """
    print(
        f"    Batch {batch_num}/{total_batches}: caricando {len(clienti_batch)} clienti..."
    )

    query = """
    UNWIND $rows AS row

    // --- 1. Ditta (CDM: Company) ---
    MERGE (di:Ditta {dittaId: row.codice_ditta})

    // --- 2. Cliente (CDM: Account, Schema.org: Organization) ---
    MERGE (c:Cliente {accountnumber: row._keycliente})
    ON CREATE SET
        c.name = row.ragione_sociale,         // Proprietà standard (CDM/Schema.org)
        c.creditlimit = CASE WHEN row.fido_euro IS NOT NULL AND row.fido_euro <> $default_numeric THEN toFloat(row.fido_euro) ELSE 0.0 END, // Proprietà CDM
        c.paymenttermscode = row.codice_pagamento, // Proprietà CDM
        c.customertypecode = row.tipoconto,        
        c.dataCaricamento = datetime()
    
    // --- 3. Relazione Cliente -> Ditta ---
    MERGE (c)-[:APPARTIENE_A]->(di)

    // --- 4. Luogo (Schema.org: Place) ---
    MERGE (l:Luogo {localita: row.localita, provincia: row.provincia, regione: row.regione, nazione: row.nazione})
    ON CREATE SET
        l.dataCaricamento = datetime()
    // Relazione Cliente -> Luogo (CDM: hasAddress)
    MERGE (c)-[:HAS_ADDRESS]->(l)
    """

    start_time = time.time()
    tx.run(query, rows=clienti_batch, default_numeric=DEFAULT_VALUES["numeric"])

    elapsed = time.time() - start_time
    print(f"    Batch {batch_num} completato in {elapsed:.2f}s")


# SOSTITUISCI L'INTERA FUNZIONE CON QUESTA
def carica_documenti_e_articoli_batch(tx, documenti_batch, batch_num, total_batches):
    print(
        f"      📦 Batch {batch_num}/{total_batches}: caricando {len(documenti_batch)} righe documento cliente..."
    )

    query = """
    UNWIND $rows AS row

    // --- 1. "GET OR CREATE" delle entità collegate ---
    MERGE (c:Cliente {accountnumber: row._keycliente})
      ON CREATE SET c.name = 'Cliente placeholder: ' + row._keycliente, c.placeholder = true

    MERGE (a:Articolo {productId: row._keyarticolo})
      ON CREATE SET a.descrizione = 'Articolo placeholder: ' + row._keyarticolo, a.placeholder = true

    MERGE (dt:DocType {codice: row.doctype_mappato.codice})
    ON CREATE SET dt.nome = row.doctype_mappato.nome

    // --- 3. MERGE sulla Testata del Documento ---
    MERGE (d:Documento {documentoId: row._keydocumento})
      ON CREATE SET d.tipoOriginale = row.tipo_documento, d.dataEmissione = date(row.data_documento)

    // --- 4. MERGE sulla Riga del Documento CON I NUOVI CAMPI STANDARD ---
    MERGE (dr:RigaDocumento {rigaId: row._keyidtidr})
      SET
        dr.quantita = toFloat(row.quantita),
        dr.importo = toFloat(row.importo),          // <-- NUOVO CAMPO
        dr.tipoValore = row.tipoValore            // <-- NUOVO CAMPO

    // --- 5. Crea tutte le relazioni ---
    MERGE (d)-[:IS_TYPE]->(dt)
    MERGE (c)-[:HA_RICEVUTO]->(d)
    MERGE (d)-[:CONTIENE_RIGA]->(dr)
    MERGE (dr)-[:RIGUARDA_ARTICOLO]->(a)
    """

    start_time = time.time()
    tx.run(query, rows=documenti_batch)
    elapsed = time.time() - start_time
    print(f"      ✅ Batch {batch_num} completato in {elapsed:.2f}s")


def carica_clienti_e_luoghi(tx, clienti):
    """
    Carica clienti e luoghi a batch per mostrare il progresso
    """
    print(f"Inizio caricamento {len(clienti)} clienti...")

    # Pulisci i record e converti i tipi
    clienti_puliti = pulisci_record_clienti(clienti)
    clienti_converted = [converti_tipi_per_neo4j(cliente) for cliente in clienti_puliti]

    print(f"Caricamento in batch di {BATCH_SIZE} record...")

    # Dividi in batch
    total_batches = (len(clienti_converted) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(clienti_converted), BATCH_SIZE):
        batch = clienti_converted[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        carica_clienti_e_luoghi_batch(tx, batch, batch_num, total_batches)


def carica_documenti_e_articoli(tx, documenti):
    """
    Carica documenti e articoli a batch per mostrare il progresso
    """
    print(f"Inizio caricamento {len(documenti)} documenti...")

    # Pulisci i record e converti i tipi
    documenti_puliti = pulisci_record_documenti(documenti)
    documenti_converted = [converti_tipi_per_neo4j(doc) for doc in documenti_puliti]

    print(f"Caricamento in batch di {BATCH_SIZE} record...")

    # Dividi in batch
    total_batches = (len(documenti_converted) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(documenti_converted), BATCH_SIZE):
        batch = documenti_converted[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        carica_documenti_e_articoli_batch(tx, batch, batch_num, total_batches)


# Indici Full-Text per il Fuzzy Matching in Neo4j
FUZZY_INDEXES_CYPHER = """
CREATE FULLTEXT INDEX clienti_fuzzy FOR (c:Cliente) ON EACH [c.name];
CREATE FULLTEXT INDEX fornitori_fuzzy FOR (gf:GruppoFornitore) ON EACH [gf.ragioneSociale];
CREATE FULLTEXT INDEX articoli_fuzzy FOR (a:Articolo) ON EACH [a.descrizione];
CREATE FULLTEXT INDEX doctype_fuzzy FOR (dt:DocType) ON EACH [dt.name];
CREATE FULLTEXT INDEX luoghi_fuzzy FOR (l:Luogo) ON EACH [l.localita];
"""


# SOSTITUISCI L'INTERA FUNZIONE CON QUESTA
def crea_indici_e_constraints(driver):
    """
    Crea indici e constraints PRIMA del caricamento per velocizzare drasticamente i MERGE.
    """
    print("\n" + "=" * 60)
    print("🚀 CREAZIONE INDICI E CONSTRAINTS")
    print("=" * 60)

    constraints_queries = [
        # --- Entità Principali ---
        "CREATE CONSTRAINT ditta_id IF NOT EXISTS FOR (d:Ditta) REQUIRE d.dittaId IS UNIQUE",
        "CREATE CONSTRAINT cliente_accountnumber IF NOT EXISTS FOR (c:Cliente) REQUIRE c.accountnumber IS UNIQUE",
        "CREATE CONSTRAINT fornitore_id IF NOT EXISTS FOR (f:Fornitore) REQUIRE f.fornitoreId IS UNIQUE",
        "CREATE CONSTRAINT gruppo_fornitore_ragione_sociale IF NOT EXISTS FOR (gf:GruppoFornitore) REQUIRE gf.ragioneSociale IS UNIQUE",
        "CREATE CONSTRAINT luogo_composite IF NOT EXISTS FOR (l:Luogo) REQUIRE (l.localita, l.provincia, l.regione, l.nazione) IS UNIQUE",
        # --- Articoli e Gerarchie (NUOVO) ---
        "CREATE CONSTRAINT articolo_productid IF NOT EXISTS FOR (a:Articolo) REQUIRE a.productId IS UNIQUE",
        # In `crea_indici_e_constraints`, modifica le righe relative a Famiglia e Sottofamiglia
        "CREATE CONSTRAINT famiglia_codice IF NOT EXISTS FOR (fam:Famiglia) REQUIRE fam.codice IS UNIQUE",
        "CREATE CONSTRAINT sottofamiglia_codice IF NOT EXISTS FOR (sfam:Sottofamiglia) REQUIRE sfam.codice IS UNIQUE",
        # --- Documenti ---
        "CREATE CONSTRAINT documento_id IF NOT EXISTS FOR (d:Documento) REQUIRE d.documentoId IS UNIQUE",
        "CREATE CONSTRAINT riga_documento_id IF NOT EXISTS FOR (dr:RigaDocumento) REQUIRE dr.rigaId IS UNIQUE",
        # La gestione DocType verrà modificata nel prossimo passo
    ]

    index_queries = [
        # Sostituisci le vecchie righe per DocType con queste:
        "CREATE CONSTRAINT doctype_codice IF NOT EXISTS FOR (dt:DocType) REQUIRE dt.codice IS UNIQUE",
        "CREATE INDEX doctype_name IF NOT EXISTS FOR (dt:DocType) ON (dt.name)",  # Lo manteniamo per ricerche testuali
        "CREATE INDEX articolo_sku IF NOT EXISTS FOR (a:Articolo) ON (a.sku)",
    ]

    with driver.session(database="neo4j") as session:
        print("📊 Creazione constraints...")
        for i, query in enumerate(constraints_queries, 1):
            try:
                session.run(query)
                print(f"   ✅ Constraint {i}/{len(constraints_queries)} creato")
            except Exception as e:
                print(f"   ⚠️  Constraint {i} già esistente o errore: {str(e)[:50]}")

        print("\n📊 Creazione indici...")
        for i, query in enumerate(index_queries, 1):
            try:
                session.run(query)
                print(f"   ✅ Indice {i}/{len(index_queries)} creato")
            except Exception as e:
                print(f"   ⚠️  Indice {i} già esistente o errore: {str(e)[:50]}")

    print("\n✅ Setup indici completato!")
    print("=" * 60 + "\n")


def main():
    """Funzione principale per eseguire l'ETL."""
    start_time = time.time()

    print(" AVVIO PROCESSO ETL DA SQL SERVER A NEO4J")
    print("=" * 60)
    print(f" Valori di default per dati mancanti:")
    print(f"   - Stringhe: '{DEFAULT_VALUES['string']}'")
    print(f"   - ID: '{DEFAULT_VALUES['id']}'")
    print(f"   - Numerici: {DEFAULT_VALUES['numeric']}")
    print(f"   - Date: '{DEFAULT_VALUES['date']}'")
    print(f"   - Dimensione batch: {BATCH_SIZE}")
    print("=" * 60)

    # 1. Connessione a Neo4j
    print("Connessione a Neo4j...")
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    try:
        # 2. Connessione a SQL Server
        crea_indici_e_constraints(neo4j_driver)
        print(" Connessione a SQL Server...")
        with pyodbc.connect(SQL_CONN_STR) as sql_conn:
            print("Connessione a MS SQL Server riuscita!")
            cursor = sql_conn.cursor()

            # 3. ESTRAZIONE DATI
            print("\n FASE DI ESTRAZIONE")
            print("-" * 40)

            print(" Estrazione clienti...")
            extraction_start = time.time()
            cursor.execute("SELECT * FROM QLK_VISTACLIENTI")
            # cursor.execute("SELECT TOP 2000 * FROM QLK_VISTACLIENTI") #per test veloce
            columns = [
                column[0].replace(" ", "_").lower() for column in cursor.description
            ]
            clienti = [dict(zip(columns, row)) for row in cursor.fetchall()]
            extraction_time = time.time() - extraction_start
            print(f" Trovati {len(clienti)} clienti in {extraction_time:.2f}s")

            # --- NUOVA ESTRAZIONE FORNITORI ---
            print("🚚 Estrazione fornitori...")
            extraction_start = time.time()
            cursor.execute("SELECT * FROM QLK_VISTAFORNITORI")
            # cursor.execute("SELECT TOP 2000 * FROM QLK_VISTAFORNITORI") #per test veloce
            columns = [
                column[0].replace(" ", "_").lower() for column in cursor.description
            ]
            fornitori = [dict(zip(columns, row)) for row in cursor.fetchall()]
            extraction_time = time.time() - extraction_start
            print(f"✅ Trovati {len(fornitori)} fornitori in {extraction_time:.2f}s")
            # --- FINE NUOVA ESTRAZIONE ---

            # --- NUOVA ESTRAZIONE ARTICOLI ---
            print("🔩 Estrazione articoli...")
            extraction_start = time.time()
            cursor.execute("SELECT * FROM QLK_VISTAARTICOLI")
            # cursor.execute("SELECT TOP 2000 * FROM QLK_VISTAARTICOLI")
            columns = [
                column[0].replace(" ", "_").lower() for column in cursor.description
            ]
            articoli = [dict(zip(columns, row)) for row in cursor.fetchall()]
            extraction_time = time.time() - extraction_start
            print(f"✅ Trovati {len(articoli)} articoli in {extraction_time:.2f}s")

            print(" Estrazione documenti...")
            extraction_start = time.time()
            cursor.execute("SELECT * FROM QLK_VISTADOCUMENTICLIENTI")
            # cursor.execute("SELECT TOP 2000 * FROM QLK_VISTADOCUMENTICLIENTI") #per test veloce
            columns = [
                column[0].replace(" ", "_").lower() for column in cursor.description
            ]
            documenti = [dict(zip(columns, row)) for row in cursor.fetchall()]
            extraction_time = time.time() - extraction_start
            print(
                f" Trovate {len(documenti)} righe di documenti in {extraction_time:.2f}s"
            )

            # --- NUOVA ESTRAZIONE DOCUMENTI FORNITORI ---
            print("🧾 Estrazione documenti fornitori...")
            extraction_start = time.time()
            cursor.execute("SELECT * FROM QLK_VISTADOCUMENTIFORNITORI")
            columns = [
                column[0].replace(" ", "_").lower() for column in cursor.description
            ]
            documenti_fornitori = [dict(zip(columns, row)) for row in cursor.fetchall()]
            extraction_time = time.time() - extraction_start
            print(
                f"✅ Trovate {len(documenti_fornitori)} righe di documenti fornitore in {extraction_time:.2f}s"
            )
            # --- FINE NUOVA ESTRAZIONE ---

            # 4. CARICAMENTO DATI IN NEO4J
            print("\n FASE DI CARICAMENTO")
            print("-" * 40)
            with neo4j_driver.session(database="neo4j") as session:

                # --- NUOVO CARICAMENTO ARTICOLI ---
                # Lo facciamo prima dei documenti per arricchire i nodi
                print("🔩 ARTICOLI")
                load_start = time.time()
                session.execute_write(carica_articoli, articoli)
                load_time = time.time() - load_start
                print(f"✅ Articoli caricati in {load_time:.2f}s\n")
                # --- FINE NUOVO CARICAMENTO ---

                print(" CLIENTI E LUOGHI")
                load_start = time.time()
                session.execute_write(carica_clienti_e_luoghi, clienti)
                load_time = time.time() - load_start
                print(f" Clienti caricati in {load_time:.2f}s")

                print("\n🚚 FORNITORI")
                load_start = time.time()
                session.execute_write(carica_fornitori, fornitori)
                load_time = time.time() - load_start
                print(f"✅ Fornitori caricati in {load_time:.2f}s")
                # --- FINE NUOVO CARICAMENTO ---

                print("\n DOCUMENTI E ARTICOLI")
                load_start = time.time()
                session.execute_write(carica_documenti_e_articoli, documenti)
                load_time = time.time() - load_start
                print(f"Documenti caricati in {load_time:.2f}s")

                # --- NUOVO CARICAMENTO DOCUMENTI FORNITORI ---
                print("\n🧾 DOCUMENTI FORNITORI (collega tutto)")
                load_start = time.time()
                session.execute_write(carica_documenti_fornitori, documenti_fornitori)
                load_time = time.time() - load_start
                print(f"✅ Documenti fornitori caricati in {load_time:.2f}s")
                # --- FINE NUOVO CARICAMENTO ---

            total_time = time.time() - start_time
            print("\n" + "=" * 60)
            print(f" PROCESSO ETL COMPLETATO CON SUCCESSO!")
            print(f" Tempo totale: {total_time:.2f}s ({total_time/60:.2f} minuti)")
            print(f" Record processati:")
            print(f"  - Clienti: {len(clienti)}")
            print(f"  - Documenti: {len(documenti)}")
            print(f"  - Fornitori: {len(fornitori)}")
            print(f"  - Documenti fornitori: {len(documenti_fornitori)}")
            print(f"  - Articoli: {len(articoli)}")
            print("\n Nota: I record con valori di default indicano dati mancanti")
            print("   La proprietà 'datiCompleti' indica la qualità dei dati")
            print("=" * 60)

    except pyodbc.Error as ex:
        sqlstate = ex.args[0] if ex.args else "Errore sconosciuto"
        print(f"ERRORE SQL SERVER: {sqlstate}")
    except ServiceUnavailable as e:
        print(f"ERRORE CONNESSIONE NEO4J: {e}")
    except Exception as e:
        print(f"ERRORE GENERICO: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "neo4j_driver" in locals():
            neo4j_driver.close()
            print("Connessione a Neo4j chiusa.")


if __name__ == "__main__":
    main()
