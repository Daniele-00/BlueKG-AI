import json
import random
import re

# ==============================================================================
# == 1. DEFINISCI LE TUE LISTE DI PLACEHOLDER (PIÙ SONO, MEGLIO È)       ==
# ==============================================================================
# Aggiungi qui nomi realistici presi dal tuo database per la massima qualità
CLIENTI = [
    "L'ABBONDANZA SRL",
    "ROSSI SPA",
    "BIANCHI & FIGLI",
    "VERDI COSTRUZIONI",
    "CLIENTE PROVA SRL",
    "ICE CONTROL SRL",
    "UMBRIAFRIGO S.R.L.",
    "CONSORZIO REAL CLEAN GROUP",
    "RIMEP SPA",
    "SIEC SRL",
    "ARRENI THE BEST SAS",
    "ARREDO SERVICE SRL",
    "ANTONELLI FRIGO",
    "MUNTEANU CONSANTIN" "ESPRINET SPA",
    "VINCENTI GATTI GIULIANO",
    "ARISTON SNC",
    "ARLECCHINI",
    "AZZURRA SNC",
    "ARCADIA CAFFE' SNC",
    "ALESPRING IMMOBILIARE SRL",
    "ALABOR SRL",
    "RN ELETTRICA di NIKOLLA ROBERT",
    "U.MATIC DI VINTI - PAGNOTTA & C.SNC",
    "SOTO VALLADARES LUIS ENRIQUE",
    "MUNTEANU CONSTANTIN",
    "FERRARO COIBENTAZIONI SRLS",
    "GANZ EINFACH GMBH",
    "MEMORYKING GMBH & CO. KG",
    "COM-FOUR VERTRIEBS GMBH",
    "CHENGDU YUEHONGSHUN SHANGMAO YOUXIAN GONGSI",
    "AMAZON EU S.A.R.L. 6",
    "SHEN ZHEN SHI HUA NUO SHI TING KE JI YOU XIAN GONG SI",
    "SHEN ZHEN SHI BEN FEI MAO YI YOU XIAN GONG SI",
    "SG IMPIANTI DI ANDREA GIUSEPPONI",
    "AMAZON EU S.A.R.L., SUCCURSALE ITALIANA",
    "SHENZHENSHIXIAOHUANDIANZISHAGWUYOUXIANGONGSI",
    "ANKER TECHNOLOGY (UK) LTD",
    "MICROSOFT IRELAND OPERATIONS LTD",
    "MICHAEL RUDEL",
    "GRUPPO CB SRL",
    "CHENG DU YI MEI JIN KE JI YOU XIAN GONG SI",
    "CHIUCHINI GIANNI",
    "STARLINK INTERNET SERVICES LIMITED",
    "TCU TRADING LTD",
    "ZYXEL NETWORKS A/S",
    "HU NAN JING WEI GONG MAO YOU XIAN GONG SI",
]
FORNITORI = [
    "CARTA CARBURANTE",
    "ITALY CHINESE KUNG FU",
    "COMPUTER GROSS SPA",
    "PRL SRL",
    "MAGAZZINI GABRIELLI SPA",
    "SHENZHEN",
    "TOTò E PEPPINO SRL",
    "FARMACIA LUCIANI SNC",
    "JAKALA SPA",
    "DEALERPOINT SRL",
    "SHENZEN SHIAIRUIERDIAN",
    "EMMA DISTRIBUZIONE SRL",
    "NOVEL FOOD SRL",
    "C.D.A. PERUGIA SRL",
    "MANZOCCHI DARIO",
    "SHENZHENSHI",
    "CR ITALY SRL",
    "SHENZHEN YING BO WEIYE",
    "NOVARREDO SRL",
    "EXERZ LIMITED",
    "WU HAN PIN SHENG KE JI YOU XIAN GONG SI",
    "KING SRL",
    "ARTGEIST sp. z o.o.",
    "SHENZHENSHIQINGTIANZHUDIANZISHANGWUYOUXIANGONGSI",
    "PAOLETTI DISTRIBUZIONE S.R.L.",
    "CHANGSHA",
    "FERRAMENTA DE FRANCESCO SAS",
    "R.M. FRIGO SNC",
    "BESTEK GLOBAL LTD",
    "10GTEK TRANSCEIVERS CO.LIMITED",
    "ZYXEL DEUTSCHLAND GMBH",
    "TECHNOENERGY SRL",
    "PESCIARELLI EMPORIO SRL",
    "TCU TRADING LTD",
    "TATKRAFT OU",
    "UNICOMP SRL",
    "DROPCASES LTD",
    "LGL REFRIGERATION ITALIA SRL",
    "MEMORYKING GMBH & CO. KG",
    "LIA E GIO'",
    "ACCESS KING BVBA",
    "DONGGUAN GUOHE MAOYI YOUXIANGONGSI",
    "SFIZIO SRL",
    "LEDSCOM.DE",
    "FUJIAN YINGHAO WENHUA CHUANGYI GUFEN YOU XIAN",
    "L'ANTICO FORZIERE SRL",
    "GUANGZHOU XUNZHE TRADING COMPANY LIMITED",
    "EURO DK SIA",
    "MANIFATTURE TECNOLEGNO HARTZ",
    "PIEFFE 92 SRL",
    "NEW DISCO SAS",
    "FERRAMENTA ALBERTI",
    "ELETTROMECCANICA B.F.P. SNC",
    "NUOVA PALLAVOLO DERUTESE",
    "MOIANO MARKET SAS",
    "ASS.TUR.PRO-LOCO P.VANNUCCI",
    "FRIGOSERVICE DI ARCANGELI MASSIMILIANO",
    "A.I.T.SRL",
    "ARTIGIANA IMPIANTI SNC",
    "CARTOLERIA ECO",
    "PHARMATEC ITALIA S.R.L.",
    "RONCHINI SRL",
    "PAPA SERENELLA & C.SNC",
    "MERICAT SRL",
]
PRODOTTI = [
    "GAS FREON SOLSTICE N40 R448A",
    "DATO_NON_DISPONIBILE",
    "INTERRUTTORE FINECORSA",
    "SERB.ACC A/REFR.LT1000",
    "SUPPORTO IN PLASTICA PER RIPIANO",
    "SPORTELLI SCORREVOLI",
    "RESISTENZA ANTICONDENSA 48V 18W CELLA",
    "COMPRESSORE A VITE V4",
    "CENTRALINA DISTRIBUZIONE",
    "FABBRICATORE DI GHIACCIO SB90A",
    "FABBRICATORE TIPO SG230A",
    "FABBRICATORE DI GHIACCIO",
    "SG350A FABBRICATORE DI GHIACCIO",
    "FABBRICATORE DI GHIACCIO MOD.S350W",
    "SG 350 W",
    "FABBRICATORE GHIACCIO S550 A",
    "FABBRICATORE SG600 AW R.01",
    "FABBRICATORE DI GHIACCIO S550W",
    "FABBRICATORE DI GHIACCIO SGS600",
    "FABBRICATORE DI GHIACCIO SP702 AW",
    "FABBRICATORE SPS701 R.01",
    "CESTONE D190",
    "DOSATORE ANTICALCARE",
    "BACINELLA IN RAME",
    "COLLARI FISSAGGIO",
    "KIT GUAR.TALLONE MAGN.C/MAG.20",
    "KIT GUAR.200X100 CON VITI",
    "PORTA A STRISCE PER CELLA",
    "CORDICELLA MM.10",
    "GUARNIZIONE PER PORTA",
    "TELAI",
    "PORTABILANCIA INOX",
    "DIVISORIO IN FILO USATO 50X9",
    "COMPR.ASPERA NB6144Z",
    "RESISTENZA RAME W1500 V220",
    "COMPRESSORE ASPERA J7240FPSC",
    "COMPRESSORE ASPERA E2134ECSR",
    "COMPR.ASPERA NE 6181 ECSIR",
    "COMPR.NB6165ECSIR",
    "COMPRESSORE ASPERA NE9213",
    "COMPR.ASPERA J6226 ECR",
    "COMPR.ASPERA NJ9232E",
    "COMPR.ASPERA T2168E",
    "COMPRESSORE ASPERA J9226P",
    "COMPR. ASPERA BP1084Z",
    "COMPRESSORE ASPERA NB2118Z",
    "COMPR.ASPERA T2134Z",
    "COMPR.ASPERA NE6170Z",
    "COMPR.ASPERA NE6187Z",
    "COMPR.ASPERA T6213Z",
    "COMPRESSORE ASPERA T6215Z",
    "COMPRESSORE NJ6220Z",
    "COMPR.ASPERA J6226Z",
    "COMPR.ASPERA NB6165GK",
    "COMPR.ASPERA NEK 6181GK",
    "COMPR.ASPERA NE6210GK",
    "COMPR.ASPERA NB6165 GK",
    "COMPR.ASPERA NEK 6213 GK",
    "COMPRESS.ASPERA T6220GK",
    "COMPRESSORE NT6220GK",
    "COMPRESSORE ASPERA T6220GK",
    "COMPR.ASPERA NT6222",
    "COMPR.ASPERA 9226GK",
    "COMPRESSORE ASPERA NJ9226GS",
    "COMPR.ASPERA NJ9232GK",
    "COMPRESSORE ASPERA NJ9238GK",
    "COMPR.ASPERA NJ9238GS",
    "COMPR.ASPERA NE2134GK",
    "COMPR.ASPERA NEK2150GK",
    "COMPRESSORE ASPERA T2168GK",
    "COMPR.ASPERA NT2178GK",
    "COMPR.ASPERA T2178GK",
    "COMPR.ASPERA T2180GK",
    "COMPR.ASPERA NT2180GK",
    "COMPR.ASPERA NT2192GK",
    "COMPRESSORE ASPERA NJ2192GK",
    "COMPR.ASPERA NJ2212GK",
    "NJ2212GS R404 ASPERA",
    "COMPR.ASPERA NB6152GK",
    "COMPR ASPERA NE9213GK",
    "COMPR.ASPERA J6220GK",
    "COMPRESSORE ELECTROLUX ML90FB",
    "COMPR.ELECTROLUX ML 45 TB",
    "COMPR.ELECTROLUX ML60TB",
    "ML80TB R404A COMPR. ELEKTROLUX",
    "COMPR.ELECTROLUX ML90TB",
    "MP 12 TB COMPR.ELECTROLUX",
    "COMPRESSORE ELECTROLUX MP 14TB",
    "COMPRESSORE ELECTROLUX MX 18TB R404A",
    "COMPRESSORE MX21TB",
    "COMPR.ELECTROLUX MS26TB",
    "COMPR.ELECTROLUX ML90FB",
    "COMPRESSORE MP14FB",
    "COMPRESSORE ELECTROLUX MX18FB",
    "MX23FB COMPR.ELECTROLUX",
    "GD 40 MB COMPRESSORE",
    "COMPR.ELECTROLUX GL 45 TB R134",
    "COMPE.ELECTROLUX GL80TB",
    "COMPR.ELECTROLUX GL90TB",
    "COMPR.ELECTROLUX GP12TB R134",
    "COMPR.ELECTROLUX GP14TB R134",
    "COMPR.ELECTROLUX GX18TB",
    "COMNPR.ELECTROLUX GS34TB",
    "COMPRESSORE ELECTROLUX L57TN",
    "COMPR.MANEUROP MTZ40JH R404",
    "COMPR.MANEUROP. MTZ40JH 4VE R404",
    "COMPR.MANEUROP SZ 84 4VI 380V R407",
    "COMPR.UN.HERMETIQUE TFH S524 E",
    "COMPR. TAJ 4519Z U.H.",
    "COMPR.UNITE' HERMETIQUE",
    "COMPR.U.H.TAG4561Z",
    "COMPR.FRASCOLD A 0.75Y",
    "A 1 6Y COMPR.FRASCOLD",
    "COMPR.FRASCOLD D213Y",
    "D 318 Y COMPRESSORE FRASCOLD",
    "COMPR-FRASCOLD D4D16Y",
    "F 4.19Y COMPRESSORE FRASCOLD",
    "COMPRESSORE FRASCOLD Q528Y",
    "COMPR.FRASCOLD S7.39Y",
    "COMPRESSORE FRASCOLD V 30.84Y",
    "COMPR.BITZER 4TC82Y40",
    "TENDINA MAN T FOR L2500",
    "CANALE CHIUSO 200 X 80",
    "COMPR.DANFOSS SC 21 CL HST",
    "COMPR.COPELAND ZR61KCE",
    "DIVISORIO VASCA 80X15H",
    "UNITA' SEM.FRASCOLD UC58-A 0.75Y",
    "UC98Z-D 2.13Y UNITA' SEM.FRASCOLD",
    "UNITA' ARIA SEMIERM.UTFA 2A0.75Y",
    "UNITA' ARIA SEMIER.UTFA 2A 1.5 7Y",
    "UNITA' COND.UTFA2B1,59.Y",
    "UNIITA' COND.SERMIERM.UTFA 2B210.1 Y-LLD",
    "UTFA 2 D 2.13.1 Y-LLD UNITA' SEMIERM",
    "UNITA' COND.UB6 144Z 02 R134",
    "UNITA' ERMETICA UE6187Z-02",
    "UNITA' ASPERA UJ2192GK",
    "UNITA' ASPERA UJ2212GS",
    "UJ9226GS UNITA' ASPERA",
    "UNITA' COND.UJ9232GS ASPERA",
    "UJ9238GS-02..2VR R404",
    "UNITA' ERMETICA UT6220GK",
    "UNITA' ASPERA UE9213GK",
    "COMPR.ASPERA UJ9226GK-02",
    "UNITA' CONDENS.ELECTROLUX GX 18 TB",
    "UNITA' COND.ERMET.TEAV 2192GK",
    "UNITA' COND.TEAV 2 J2212GK",
    "RUOTA",
    "UNITA' CONDENSATRICE TEAV NEK6165GK",
    "UNITA' COND.ERMET.TEAV NEK6181GK",
    "UNITA' COND.ERM.TEAV NEK6213GK",
    "UNITA' COND.ERMET.TEAV T 6220GK",
    "UNITA' COND.ERMET.TEAV2 T6220GK",
    "UNITA' COND.ERM.TEAV J9226GK",
    "UNITA' COND.ERMET. TEAV 2 NT6226GK",
    "UNITA'COND.ERMET.TEAV2 J9238GK",
    "UNITA' COND.ERMET. TEAC NB6152GK",
    "UNITA' COND. ERMET.TEAC 6165GK",
    "UNITAA' COND.ERTM.TEAV NE6187Z",
    "UNITA' COND.ERMET.TEAV J6220Z",
    "UNITA' COND.ERMET.TEAV 2J2192GS",
    "UNITA' CARENATA UTFACA D 11.1",
    "UNITA' CARENATA UTFACA D 313.1Y",
    "UNITA' CARENATA SIL.UTFACS D 313.1Y",
    "UNITA' CARENATA UTFACALC 2 B 1,59.1Y-LLD",
    "UNITA' CONDENSATRICE DORIN H300CC",
    "unita' condensatrice uchg 33 a nek6210gk",
    "UNITA' CONDENSATRICE TUC HG 150 A/2",
    "UNITA' CONDENSATRICE",
]
DITTE_instruction = ["1", "2", "3", "4"]
LOCALITA = ["Perugia", "Milano", "Roma", "Napoli", "Torino"]
PROVINCE = ["PG", "MI", "RM", "NA", "TO"]
FAMIGLIE = [
    "CENTRALI FRIGO",
    "N/D",
    "MOTOCOMPRESSORI",
    "BACINELLA",
    "RESISTENZE",
    "MOTOVENTILATORI",
    "CONDENSATORI",
    "EVAPORATORI",
    "ARMAFLEX",
    "TERMOSTATI",
    "VALVOLE",
    "CENTRALI FRIGO",
    "FIANCO",
    "TUBI NEON",
    "CRISTALLI SAGOMATI",
    "TEMPORIZZATORI",
    "INTERRUTTORI",
    "RAME",
    "BANCO FRIGO",
    "TENDINE",
    "GRUPPI FRIGO",
    "FRONTALE",
]
ANNI = ["2021", "2022", "2023", "2024", "2025"]
MESI = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
TOP_N_VALUES = [3, 5, 10, 20, 50, 100]
CODICI_DOC = [
    "DDT C/VENDITA",
    "FATT. RIEPILOGATIVA CLIENTE",
    "FATT. IMMEDIATA VENDITA",
    "NOTA DI CREDITO VENDITA",
    "NOTA DI CREDITO S/MAGAZZINO",
    "DDT INSTALLAZIONE PERUGIA",
    "DDT C/VENDITA ROMA",
    "DDT C/VENDITA PERUGIA",
    "PREFATTURAZION CONTRATTI ROMA",
    "DDT INSTALLAZIONE ROMA",
    "PREFATTURAZION CONTRATTI PERUGIA",
    "DDT RESO C/LAVORO PERUGIA",
    "DDT C/RIPARAZIONE DA CLIENTE",
    "DDT RESO C/LAVORO ROMA",
    "DDT C/VISIONE A CLIENTE",
    "DDT A CLIENTE NF",
    "DDT C/RIP RITIRO BENE DA RIPARARE RM",
    "FATT. RIEPILOGATIVA CLIENTE ROMA",
    "FATT. RIEPILOGATIVA CLIENTE PERUGIA",
    "FATT. IMMEDIATA VENDITA ROMA",
    "FATT. IMMEDIATA VENDITA PERUGIA",
    "FATTURA D'ACCONTO PERUGIA",
    "Fattura vendita con ritenute UF",
    "FATTURA D'ACCONTO ROMA",
    "FATT. IMMEDIATA VENDITA PROVA MAG",
    "NOTA DI CREDITO S/MAGAZZINO ROMA",
    "NOTA DI CREDITO S/MAGAZZINO PERUGIA",
    "NOTA DI CREDITO VENDITA ROMA",
    "NOTA DI CREDITO VENDITA PERUGIA",
    "NOTA DEBITO VENDITA ROMA",
    "NOTA DEBITO VENDITA PERUGIA",
    "PREVENTIVO CLIENTE - PERUGIA",
    "PREVENTIVO CLIENTE - ROMA",
    "FATT. AUTOCONSUMO PERUGIA",
    "ORDINE DA CLIENTE - ROMA",
    "ORDINE DA CLIENTE - PERUGIA",
    "DDT FOGLI DI LAVORO",
    "DDT INSTALLAZIONE",
    "FATT. IMMEDIATA VENDITA TD26",
    "DDT RESO A FORNITORE",
    "Fattura (autofattura) TD17 senza Coge",
    "ORDINE A FORNITORE",
    "Fattura (autofattura) TD18 senza Coge",
    "Fattura (autofattura) TD16 senza Coge",
    "Fattura immediata acquisto no coge",
    "Fattura immediata acquisto sp",
    "ORDINE A FORNITORE PERUGIA",
    "RICHIESTA PREV. FORNITORE PER ACQUISTO",
    "ORDINE A FORNITORE ROMA",
    "DDT DI ACQUISTO PERUGIA",
    "DDT RESO A FORNITORE PERUGIA",
    "DDT C/RIPARAZIONE A FORNITORI",
    "DDT DI ACQUISTO ROMA",
    "DDT RESO C/RIPARAZIONE DA FORNITORI",
    "DDT RESO A FORNITORE ROMA",
]

TIPO_VALORE_VARIANTS = [
    {"keyword": "ordinato", "value": "Ordinato"},
    {"keyword": "bollato", "value": "Bollato"},
    {"keyword": "fatturato", "value": "Fatturato"},
]

DOCTYPE_VARIANTS = [
    # Ciclo Attivo (Cliente)
    {"user_term": "fatture cliente", "code": "FVC", "cycle": "CLIENTE"},
    {"user_term": "ordini cliente", "code": "OC", "cycle": "CLIENTE"},
    {"user_term": "DDT cliente", "code": "DDTV", "cycle": "CLIENTE"},
    {"user_term": "note di credito cliente", "code": "NCC", "cycle": "CLIENTE"},
    # Ciclo Passivo (Fornitore)
    {"user_term": "fatture fornitore", "code": "FFA", "cycle": "FORNITORE"},
    {"user_term": "ordini a fornitore", "code": "OF", "cycle": "FORNITORE"},
    {"user_term": "DDT di acquisto", "code": "DDTA", "cycle": "FORNITORE"},
]


# ==============================================================================
# == 2. INCOLLA QUI I TUOI ESEMPI DI BASE (IL "GOLDEN SET")              ==
# ==============================================================================
# Ho unito tutti i tuoi esempi in un'unica lista
base_examples = [
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è il fatturato totale del cliente BARONE DI FERRANTE EZIO?",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE toLower(trim(c.name)) = 'barone di ferrante ezio' AND dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN sum(dr.importo) AS fatturatoTotale",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Chi è il cliente con il maggior fatturato?",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN c.name AS cliente, sum(dr.importo) AS fatturatoTotale ORDER BY fatturatoTotale DESC LIMIT 1",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Elenca 5 clienti che si trovano a Perugia",
        "output": "MATCH (c:Cliente)-[:HAS_ADDRESS]->(l:Luogo) WHERE toLower(trim(l.localita)) = 'perugia' RETURN DISTINCT c.name AS cliente ORDER BY cliente LIMIT 5",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Mostrami i 5 prodotti più popolari in termini di quantità venduta",
        "output": "MATCH (:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) RETURN a.descrizione AS prodotto, sum(dr.quantita) AS quantitaTotale ORDER BY quantitaTotale DESC LIMIT 5",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Dimmi il fatturato di ogni ditta.",
        "output": "MATCH (d:Ditta)<-[:APPARTIENE_A]-(:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN d.dittaId AS ditta, sum(dr.importo) AS fatturatoTotale ORDER BY fatturatoTotale DESC",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Il fatturato di ogni ditta, nel loro anno migliore",
        "output": "MATCH (d:Ditta)<-[:APPARTIENE_A]-(:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 WITH d, doc.dataEmissione.year AS anno, sum(dr.importo) AS fatturatoAnnuale ORDER BY d.dittaId, fatturatoAnnuale DESC WITH d, collect({anno: anno, fatturato: fatturatoAnnuale}) AS fatturatiOrdinati RETURN d.dittaId AS ditta, fatturatiOrdinati[0].anno AS annoTop, fatturatiOrdinati[0].fatturato AS fatturatoMassimo",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è la famiglia di prodotti più acquistata?",
        "output": "MATCH (:Fornitore)-[:HA_EMESSO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo)-[:APPARTIENE_A]->(sf:Sottofamiglia)-[:INCLUSA_IN]->(fam:Famiglia) RETURN fam.nome AS famiglia, sum(dr.quantita) AS quantitaTotale ORDER BY quantitaTotale DESC LIMIT 1",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è il prodotto più venduto in generale (per valore)?",
        "output": "MATCH (:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN a.descrizione AS prodotto, sum(dr.importo) AS valoreVenduto ORDER BY valoreVenduto DESC LIMIT 1",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "A quanto ho acquistato il prodotto GAS FREON SOLSTICE N40 R448A?",
        "output": "MATCH (a:Articolo)<-[:RIGUARDA_ARTICOLO]-(dr:RigaDocumento)<-[:CONTIENE_RIGA]-(:Documento)<-[:HA_EMESSO]-(:Fornitore) WHERE toLower(trim(a.descrizione)) = 'gas freon solstice n40 r448a' RETURN sum(dr.importo) AS costoTotaleAcquisto",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "A quanto l'ho venduto il prodotto GAS FREON SOLSTICE N40 R448A?",
        "output": "MATCH (:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) WHERE toLower(trim(a.descrizione)) = 'gas freon solstice n40 r448a' AND dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN sum(dr.importo) AS totaleVenduto",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Attualmente la ditta 2 è in perdita o in crescita?",
        "output": "MATCH (d:Ditta {dittaId: '2'})<-[:APPARTIENE_A]-(:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 AND doc.dataEmissione.year IN [date().year, date().year - 1] RETURN doc.dataEmissione.year AS anno, sum(dr.importo) AS fatturatoAnnuale ORDER BY anno DESC",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quali ditte vendono prodotti della famiglia 'CENTRALI FRIGO'?",
        "output": "MATCH (d:Ditta)<-[:APPARTIENE_A]-(:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo)-[:APPARTIENE_A]->(:Sottofamiglia)-[:INCLUSA_IN]->(fam:Famiglia) WHERE toLower(trim(fam.nome)) = 'centrali frigo' RETURN DISTINCT d.dittaId AS ditta ORDER BY ditta",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quanti clienti unici ha la ditta '1'?",
        "output": "MATCH (d:Ditta {dittaId: '1'})<-[:APPARTIENE_A]-(c:Cliente) RETURN count(DISTINCT c) AS numeroClientiUnici",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quale fornitore ci ha inviato più documenti?",
        "output": "MATCH (f:Fornitore)-[:HA_EMESSO]->(doc:Documento) RETURN f.ragioneSociale AS fornitore, count(doc) AS numeroDocumenti ORDER BY numeroDocumenti DESC LIMIT 1",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Elenca 5 clienti che non hanno mai acquistato il prodotto 'GAS FREON SOLSTICE N40 R448A'",
        "output": "MATCH (a:Articolo) WHERE toLower(trim(a.descrizione)) = 'gas freon solstice n40 r448a' WITH a MATCH (c:Cliente) WHERE NOT (c)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a) RETURN c.name AS cliente ORDER BY cliente LIMIT 5",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è la località con il maggior numero di fornitori?",
        "output": "MATCH (f:Fornitore)-[:SI_TROVA_A]->(l:Luogo) WHERE l.localita IS NOT NULL RETURN l.localita AS localita, count(DISTINCT f) AS numeroFornitori ORDER BY numeroFornitori DESC LIMIT 1",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è l'importo medio per riga documento nelle fatture di vendita?",
        "output": "MATCH (:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN avg(dr.importo) AS importoMedioRiga",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Mostrami 10 documenti di tipo 'FVC' emessi a Gennaio 2025 al cliente 'L'ABBONDANZA SRL'",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:IS_TYPE]->(dt:DocType) WHERE toLower(trim(c.name)) = 'l\\'abbondanza srl' AND dt.codice = 'FVC' AND doc.dataEmissione.year = 2025 AND doc.dataEmissione.month = 1 RETURN doc.documentoinstruction AS documento LIMIT 10",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è il margine di profitto del prodotto 'GAS FREON SOLSTICE N40 R448A'?",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr1:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) WHERE toLower(trim(a.descrizione)) = 'gas freon solstice n40 r448a' AND dr1.tipoValore = 'Fatturato' AND dr1.importo > 0 WITH a, sum(dr1.importo) AS fatturato MATCH (f:Fornitore)-[:HA_EMESSO]->(:Documento)-[:CONTIENE_RIGA]->(dr2:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a) WITH a, fatturato, sum(dr2.importo) AS costo RETURN a.descrizione AS prodotto, fatturato, costo, (fatturato - costo) AS margine",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quali fornitori ci forniscono prodotti venduti al cliente 'BARONE DI FERRANTE EZIO'?",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) WHERE toLower(trim(c.name)) = 'barone di ferrante ezio' WITH collect(DISTINCT a) AS articoliVenduti MATCH (gf:GruppoFornitore)<-[:RAGGRUPPATO_SOTTO]-(:Fornitore)-[:HA_EMESSO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a) WHERE a IN articoliVenduti RETURN DISTINCT gf.ragioneSociale AS fornitore ORDER BY fornitore",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Elenca 5 prodotti che abbiamo venduto ma mai acquistato",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) WHERE NOT EXISTS { MATCH (a)<-[:RIGUARDA_ARTICOLO]-(:RigaDocumento)<-[:CONTIENE_RIGA]-(:Documento)<-[:HA_EMESSO]-(:Fornitore) } RETURN DISTINCT a.descrizione AS prodotto ORDER BY prodotto LIMIT 5",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Elenca 5 prodotti che abbiamo acquistato ma mai venduto",
        "output": "MATCH (f:Fornitore)-[:HA_EMESSO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) WHERE NOT EXISTS { MATCH (a)<-[:RIGUARDA_ARTICOLO]-(:RigaDocumento)<-[:CONTIENE_RIGA]-(:Documento)<-[:HA_RICEVUTO]-(:Cliente) } RETURN DISTINCT a.descrizione AS prodotto ORDER BY prodotto LIMIT 5",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è il fatturato della sottofamiglia 'COMPRESSORI SEMIERMETICI'?",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(:Articolo)-[:APPARTIENE_A]->(sf:Sottofamiglia) WHERE toLower(trim(sf.nome)) = 'compressori semiermetici' AND dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN sum(dr.importo) AS fatturatoSottofamiglia",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Elenca le sottofamiglie della famiglia 'CENTRALI FRIGO'",
        "output": "MATCH (f:Famiglia)<-[:INCLUSA_IN]-(sf:Sottofamiglia) WHERE toLower(trim(f.nome)) = 'centrali frigo' RETURN DISTINCT sf.nome AS sottofamiglia ORDER BY sottofamiglia",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è il fatturato del primo trimestre 2024?",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 AND doc.dataEmissione.year = 2024 AND doc.dataEmissione.month >= 1 AND doc.dataEmissione.month <= 3 RETURN sum(dr.importo) AS fatturatoTrimestre",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è il fatturato tra Gennaio 2024 e Marzo 2024?",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 AND doc.dataEmissione >= date('2024-01-01') AND doc.dataEmissione <= date('2024-03-31') RETURN sum(dr.importo) AS fatturatoPeriodo",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Elenca i clienti con fatturato superiore a 50000 euro",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 WITH c, sum(dr.importo) AS fatturatoTotale WHERE fatturatoTotale > 50000 RETURN c.name AS cliente, fatturatoTotale ORDER BY fatturatoTotale DESC",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quali sono i 3 prodotti più venduti al cliente 'BARONE DI FERRANTE EZIO'?",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) WHERE toLower(trim(c.name)) = 'barone di ferrante ezio' AND dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN a.descrizione AS prodotto, sum(dr.importo) AS totaleVenduto ORDER BY totaleVenduto DESC LIMIT 3",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quanti fornitori appartengono a ciascun gruppo di fornitori?",
        "output": "MATCH (gf:GruppoFornitore)<-[:RAGGRUPPATO_SOTTO]-(f:Fornitore) RETURN gf.ragioneSociale AS gruppo, count(f) AS numeroFornitori ORDER BY numeroFornitori DESC",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è l'importo minimo registrato in una singola riga di fattura di vendita?",
        "output": "MATCH (dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN min(dr.importo) AS importoMinimo",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è la quantità media di articoli per riga nei documenti di acquisto?",
        "output": "MATCH (:Fornitore)-[:HA_EMESSO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) RETURN avg(dr.quantita) AS quantitaMediaAcquisto",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quanti articoli unici sono presenti nel catalogo?",
        "output": "MATCH (a:Articolo) RETURN count(DISTINCT a) AS totaleArticoli",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Elenca i clienti dell'azienda '1' situati nella provincia di 'PG'",
        "output": "MATCH (d:Ditta {dittaId: '1'})<-[:APPARTIENE_A]-(c:Cliente)-[:HAS_ADDRESS]->(l:Luogo) WHERE toLower(trim(l.provincia)) = 'pg' RETURN c.name AS cliente ORDER BY cliente",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Calcola il fatturato totale per il secondo trimestre del 2024 (aprile, maggio, giugno).",
        "output": "MATCH (:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 AND doc.dataEmissione.year = 2024 AND doc.dataEmissione.month IN [4, 5, 6] RETURN sum(dr.importo) AS fatturatoTrimestre",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Trova tutti gli articoli la cui descrizione inizia con 'TUBO'",
        "output": "MATCH (a:Articolo) WHERE toLower(trim(a.descrizione)) STARTS WITH 'tubo' RETURN a.descrizione AS articolo",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Mostrami i documenti di tipo 'FNA' ricevuti dal fornitore 'FORNITORE ESEMPIO SPA'",
        "output": "MATCH (gf:GruppoFornitore)<-[:RAGGRUPPATO_SOTTO]-(:Fornitore)-[:HA_EMESSO]->(doc:Documento)-[:IS_TYPE]->(dt:DocType) WHERE toLower(trim(gf.ragioneSociale)) = 'fornitore esempio spa' AND dt.codice = 'FNA' RETURN doc.documentoinstruction AS documento",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Chi sono i 3 clienti con il fatturato più basso?",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN c.name AS cliente, sum(dr.importo) AS fatturatoTotale ORDER BY fatturatoTotale ASC LIMIT 3",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quali sono i 5 articoli con il costo totale di acquisto più alto?",
        "output": "MATCH (a:Articolo)<-[:RIGUARDA_ARTICOLO]-(dr:RigaDocumento)<-[:CONTIENE_RIGA]-(:Documento)<-[:HA_EMESSO]-(:Fornitore) RETURN a.descrizione AS articolo, sum(dr.importo) AS costoTotale ORDER BY costoTotale DESC LIMIT 5",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quale fornitore ci vende la gamma più ampia di articoli unici?",
        "output": "MATCH (gf:GruppoFornitore)<-[:RAGGRUPPATO_SOTTO]-(:Fornitore)-[:HA_EMESSO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) RETURN gf.ragioneSociale AS fornitore, count(DISTINCT a) AS varietaProdotti ORDER BY varietaProdotti DESC LIMIT 1",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Ordina le aziende in base al numero totale di documenti di vendita emessi.",
        "output": "MATCH (d:Ditta)<-[:APPARTIENE_A]-(:Cliente)-[:HA_RICEVUTO]->(doc:Documento) RETURN d.dittaId AS ditta, count(doc) AS numeroDocumenti ORDER BY numeroDocumenti DESC",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Elenca i clienti che hanno acquistato prodotti forniti da fornitori della stessa provincia del cliente.",
        "output": "MATCH (c:Cliente)-[:HAS_ADDRESS]->(l:Luogo) WITH c, l.provincia AS provinciaCliente MATCH (gf:GruppoFornitore)<-[:RAGGRUPPATO_SOTTO]-(:Fornitore)-[:SI_TROVA_A]->(lf:Luogo) WHERE lf.provincia = provinciaCliente WITH c, collect(DISTINCT gf) AS fornitoriLocali MATCH (c)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) MATCH (a)<-[:RIGUARDA_ARTICOLO]-(:RigaDocumento)<-[:CONTIENE_RIGA]-(:Documento)<-[:HA_EMESSO]-(:Fornitore)-[:RAGGRUPPATO_SOTTO]->(gf_check:GruppoFornitore) WHERE gf_check IN fornitoriLocali RETURN DISTINCT c.name AS cliente",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Chi sono i principali fornitori degli articoli venduti al nostro miglior cliente?",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 WITH c, sum(dr.importo) AS fatturato ORDER BY fatturato DESC LIMIT 1 MATCH (c)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) WITH collect(DISTINCT a) AS articoliTop MATCH (gf:GruppoFornitore)<-[:RAGGRUPPATO_SOTTO]-(:Fornitore)-[:HA_EMESSO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a_check:Articolo) WHERE a_check IN articoliTop RETURN gf.ragioneSociale as fornitore, sum(dr.importo) AS valoreAcquistato ORDER BY valoreAcquistato DESC",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quali sono le famiglie di prodotti vendute sia dalla ditta '1' che dalla ditta '2'?",
        "output": "MATCH (d1:Ditta {ditta: '1'})<-[:APPARTIENE_A]-(:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(:Articolo)-[:APPARTIENE_A]->(:Sottofamiglia)-[:INCLUSA_IN]->(fam:Famiglia) WITH collect(DISTINCT fam) AS famiglieDitta1 MATCH (d2:Ditta {ditta: '2'})<-[:APPARTIENE_A]-(:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(:Articolo)-[:APPARTIENE_A]->(:Sottofamiglia)-[:INCLUSA_IN]->(fam_check:Famiglia) WHERE fam_check IN famiglieDitta1 RETURN fam_check.nome AS famigliaComune",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Mostrami 5 prodotti che abbiamo acquistato ma mai venduto.",
        "output": "MATCH (a:Articolo)<-[:RIGUARDA_ARTICOLO]-(:RigaDocumento)<-[:CONTIENE_RIGA]-(:Documento)<-[:HA_EMESSO]-(:Fornitore) WITH DISTINCT a WHERE NOT (a)<-[:RIGUARDA_ARTICOLO]-(:RigaDocumento {tipoValore: 'Fatturato'}) RETURN a.descrizione AS prodottoMaiVenduto LIMIT 5",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Mostrami il fatturato totale per ogni mese dell'anno corrente.",
        "output": "MATCH (:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 AND doc.dataEmissione.year = date().year RETURN doc.dataEmissione.month AS mese, sum(dr.importo) AS fatturatoMensile ORDER BY mese",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Confronta il fatturato totale del 2024 con quello del 2023.",
        "output": "MATCH (:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 AND doc.dataEmissione.year IN [2023, 2024] RETURN doc.dataEmissione.year AS anno, sum(dr.importo) AS fatturatoTotale ORDER BY anno",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quale fornitore è stato il più attivo (per valore di acquisto) nell'ultimo trimestre?",
        "output": "WITH date() - duration({months: 3}) AS dataInizioTrimestre MATCH (gf:GruppoFornitore)<-[:RAGGRUPPATO_SOTTO]-(:Fornitore)-[:HA_EMESSO]->(doc:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE doc.dataEmissione >= dataInizioTrimestre RETURN gf.ragioneSociale AS fornitore, sum(dr.importo) AS totaleAcquistato ORDER BY totaleAcquistato DESC LIMIT 1",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Mostrami i clienti il cui primo documento di vendita è stato emesso negli ultimi 6 mesi.",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(doc:Documento) WITH c, min(doc.dataEmissione) AS primoAcquisto WHERE primoAcquisto >= date() - duration({months: 6}) RETURN c.name AS clienteRecente, primoAcquisto ORDER BY primoAcquisto DESC",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Ci sono aziende che non hanno registrato alcun fatturato?",
        "output": "MATCH (d:Ditta) WHERE NOT (d)<-[:APPARTIENE_A]-(:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento {tipoValore: 'Fatturato'}) RETURN d.dittaId AS dittaSenzaFatturato",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Trova i fornitori che non ci hanno mai venduto prodotti della famiglia 'CENTRALI FRIGO'",
        "output": "MATCH (fam:Famiglia) WHERE toLower(trim(fam.nome)) = 'centrali frigo' WITH fam MATCH (gf:GruppoFornitore) WHERE NOT (gf)<-[:RAGGRUPPATO_SOTTO]-(:Fornitore)-[:HA_EMESSO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(:Articolo)-[:APPARTIENE_A]->(:Sottofamiglia)-[:INCLUSA_IN]->(fam) RETURN gf.ragioneSociale AS fornitore",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quali clienti appartengono esclusivamente all'azienda '1' e non ad altre?",
        "output": "MATCH (c:Cliente)-[:APPARTIENE_A]->(d:Ditta) WITH c, collect(d.dittaId) AS ditte WHERE size(ditte) = 1 AND ditte[0] = '1' RETURN c.name AS clienteEsclusivo",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "In quali anni non abbiamo acquistato nulla dal fornitore 'FORNITORE ESEMPIO SPA'?",
        "output": "MATCH (doc:Documento) WITH collect(DISTINCT doc.dataEmissione.year) AS anniConMovimenti MATCH (gf:GruppoFornitore) WHERE toLower(trim(gf.ragioneSociale)) = 'fornitore esempio spa' MATCH (gf)<-[:RAGGRUPPATO_SOTTO]-(:Fornitore)-[:HA_EMESSO]->(doc_f:Documento) WITH anniConMovimenti, collect(DISTINCT doc_f.dataEmissione.year) AS anniFornitore CALL apoc.coll.subtract(anniConMovimenti, anniFornitore) YIELD value RETURN value AS annoSenzaAcquisti",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Per ogni famiglia di prodotto, chi è il cliente che ha generato il fatturato maggiore?",
        "output": "MATCH (fam:Famiglia)<-[:INCLUSA_IN]-(:Sottofamiglia)<-[:APPARTIENE_A]-(:Articolo)<-[:RIGUARDA_ARTICOLO]-(dr:RigaDocumento)<-[:CONTIENE_RIGA]-(:Documento)<-[:HA_RICEVUTO]-(c:Cliente) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 WITH fam, c, sum(dr.importo) AS fatturato ORDER BY fam.nome, fatturato DESC WITH fam, collect({cliente: c.name, fatturato: fatturato}) AS clientiOrdinati RETURN fam.nome AS famiglia, clientiOrdinati[0].cliente AS topCliente, clientiOrdinati[0].fatturato AS fatturatoMassimo",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Calcola il rapporto tra costo totale e fatturato totale per il prodotto 'GAS FREON SOLSTICE N40 R448A'",
        "output": "MATCH (a:Articolo) WHERE toLower(trim(a.descrizione)) = 'gas freon solstice n40 r448a' OPTIONAL MATCH (a)<-[:RIGUARDA_ARTICOLO]-(dr_acq:RigaDocumento)<-[:CONTIENE_RIGA]-(:Documento)<-[:HA_EMESSO]-(:Fornitore) WITH a, sum(dr_acq.importo) AS costoTotale OPTIONAL MATCH (a)<-[:RIGUARDA_ARTICOLO]-(dr_ven:RigaDocumento {tipoValore: 'Fatturato'})<-[:CONTIENE_RIGA]-(:Documento)<-[:HA_RICEVUTO]-(:Cliente) WHERE dr_ven.importo > 0 WITH costoTotale, sum(dr_ven.importo) AS fatturatoTotale RETURN costoTotale, fatturatoTotale, (fatturatoTotale / costoTotale) AS rapporto",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quale provincia ha generato il volume di fatturato più alto?",
        "output": "MATCH (c:Cliente)-[:HAS_ADDRESS]->(l:Luogo) MATCH (c)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN l.provincia AS provincia, sum(dr.importo) AS fatturatoTotale ORDER BY fatturatoTotale DESC LIMIT 1",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Lista dei clienti il cui fatturato totale è superiore al fatturato medio di tutti i clienti.",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 WITH count(DISTINCT c) AS totalCustomers, sum(dr.importo) AS totalRevenue WITH totalRevenue / totalCustomers AS avgRevenue MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 WITH c, sum(dr.importo) AS customerRevenue, avgRevenue WHERE customerRevenue > avgRevenue RETURN c.name AS cliente, customerRevenue ORDER BY customerRevenue DESC",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Mostrami il numero di documenti per ciascun tipo (codice).",
        "output": "MATCH (doc:Documento)-[:IS_TYPE]->(dt:DocType) RETURN dt.codice AS tipoDocumento, count(doc) AS numeroDocumenti ORDER BY numeroDocumenti DESC",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è il fatturato totale associato ai documenti di tipo 'FVC'?",
        "output": "MATCH (doc:Documento)-[:IS_TYPE]->(dt:DocType {codice: 'FVC'}) MATCH (doc)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN sum(dr.importo) AS fatturatoFVC",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quale cliente ha ricevuto il maggior numero di note di credito (codice 'NCC')?",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:IS_TYPE]->(dt:DocType {codice: 'NCC'}) RETURN c.name AS cliente, count(doc) AS numeroNoteCredito ORDER BY numeroNoteCredito DESC LIMIT 1",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Calcola il valore medio di un documento per ciascun tipo (considerando solo il fatturato).",
        "output": "MATCH (doc:Documento)-[:IS_TYPE]->(dt:DocType) MATCH (doc)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 WITH dt, doc, sum(dr.importo) AS valoreDocumento RETURN dt.codice AS tipoDocumento, avg(valoreDocumento) AS valoreMedio ORDER BY valoreMedio DESC",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Chi è il cliente con il fatturato più alto?",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN c.name AS cliente, sum(dr.importo) AS fatturatoTotale ORDER BY fatturatoTotale DESC LIMIT 1",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Elenca 5 clienti situati a Perugia",
        "output": "MATCH (c:Cliente)-[:HAS_ADDRESS]->(l:Luogo) WHERE toLower(trim(l.localita)) = 'perugia' RETURN DISTINCT c.name AS cliente ORDER BY cliente LIMIT 5",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Mostrami i 5 prodotti più popolari in termini di quantità venduta",
        "output": "MATCH (:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) RETURN a.descrizione AS prodotto, sum(dr.quantita) AS quantitaTotale ORDER BY quantitaTotale DESC LIMIT 5",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Dimmi il fatturato di ciascuna azienda.",
        "output": "MATCH (d:Ditta)<-[:APPARTIENE_A]-(:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN d.dittaId AS ditta, sum(dr.importo) AS fatturatoTotale ORDER BY fatturatoTotale DESC",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Il fatturato di ciascuna azienda, nel loro anno migliore",
        "output": "MATCH (d:Ditta)<-[:APPARTIENE_A]-(:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 WITH d, doc.dataEmissione.year AS anno, sum(dr.importo) AS fatturatoAnnuale ORDER BY d.dittaId, fatturatoAnnuale DESC WITH d, collect({anno: anno, fatturato: fatturatoAnnuale}) AS fatturatiOrdinati RETURN d.dittaId AS ditta, fatturatiOrdinati[0].anno AS annoTop, fatturatiOrdinati[0].fatturato AS fatturatoMassimo",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è la famiglia di prodotti più acquistata?",
        "output": "MATCH (:Fornitore)-[:HA_EMESSO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo)-[:APPARTIENE_A]->(sf:Sottofamiglia)-[:INCLUSA_IN]->(fam:Famiglia) RETURN fam.nome AS famiglia, sum(dr.quantita) AS quantitaTotale ORDER BY quantitaTotale DESC LIMIT 1",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è il prodotto più venduto in assoluto (per valore)?",
        "output": "MATCH (:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN a.descrizione AS prodotto, sum(dr.importo) AS valoreVenduto ORDER BY valoreVenduto DESC LIMIT 1",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quanto ho speso per acquistare il prodotto GAS FREON SOLSTICE N40 R448A?",
        "output": "MATCH (a:Articolo)<-[:RIGUARDA_ARTICOLO]-(dr:RigaDocumento)<-[:CONTIENE_RIGA]-(:Documento)<-[:HA_EMESSO]-(:Fornitore) WHERE toLower(trim(a.descrizione)) = 'gas freon solstice n40 r448a' RETURN sum(dr.importo) AS costoTotaleAcquisto",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quanto ho guadagnato vendendo il prodotto GAS FREON SOLSTICE N40 R448A?",
        "output": "MATCH (:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) WHERE toLower(trim(a.descrizione)) = 'gas freon solstice n40 r448a' AND dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN sum(dr.importo) AS totaleVenduto",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "L'azienda 2 sta attualmente perdendo o crescendo?",
        "output": "MATCH (d:Ditta {dittaId: '2'})<-[:APPARTIENE_A]-(:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 AND doc.dataEmissione.year IN [date().year, date().year - 1] RETURN doc.dataEmissione.year AS anno, sum(dr.importo) AS fatturatoAnnuale ORDER BY anno DESC",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quali aziende vendono prodotti della famiglia 'CENTRALI FRIGO'?",
        "output": "MATCH (d:Ditta)<-[:APPARTIENE_A]-(:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo)-[:APPARTIENE_A]->(:Sottofamiglia)-[:INCLUSA_IN]->(fam:Famiglia) WHERE toLower(trim(fam.nome)) = 'centrali frigo' RETURN DISTINCT d.dittaId AS ditta ORDER BY ditta",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quanti clienti unici ha l'azienda '1'?",
        "output": "MATCH (d:Ditta {dittaId: '1'})<-[:APPARTIENE_A]-(c:Cliente) RETURN count(DISTINCT c) AS numeroClientiUnici",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quale fornitore ci ha inviato il maggior numero di documenti?",
        "output": "MATCH (gf:GruppoFornitore)<-[:RAGGRUPPATO_SOTTO]-(:Fornitore)-[:HA_EMESSO]->(doc:Documento) RETURN gf.ragioneSociale AS fornitore, count(doc) AS numeroDocumenti ORDER BY numeroDocumenti DESC LIMIT 1",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Elenca 5 clienti che non hanno mai acquistato il prodotto 'GAS FREON SOLSTICE N40 R448A'",
        "output": "MATCH (a:Articolo) WHERE toLower(trim(a.descrizione)) = 'gas freon solstice n40 r448a' WITH a MATCH (c:Cliente) WHERE NOT (c)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a) RETURN c.name AS cliente ORDER BY cliente LIMIT 5",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quale località ha il maggior numero di fornitori?",
        "output": "MATCH (f:Fornitore)-[:SI_TROVA_A]->(l:Luogo) WHERE l.localita IS NOT NULL RETURN l.localita AS localita, count(DISTINCT f) AS numeroFornitori ORDER BY numeroFornitori DESC LIMIT 1",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è l'importo medio per riga documento nelle fatture di vendita?",
        "output": "MATCH (:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN avg(dr.importo) AS importoMedioRiga",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Mostrami i documenti di tipo 'FVC' emessi a gennaio 2025 per il cliente 'L'ABBONDANZA SRL'",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:IS_TYPE]->(dt:DocType) WHERE toLower(trim(c.name)) = 'l\\'abbondanza srl' AND dt.codice = 'FVC' AND doc.dataEmissione.year = 2025 AND doc.dataEmissione.month = 1 RETURN doc.documentoinstruction AS documento",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Per ogni azienda, classifica il suo fatturato totale come 'Alto' se supera i 10 milioni, 'Medio' se è tra 1 e 10 milioni, e 'Basso' altrimenti.",
        "output": "MATCH (d:Ditta)<-[:APPARTIENE_A]-(:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 WITH d, sum(dr.importo) AS fatturatoTotale RETURN d.dittaId AS ditta, fatturatoTotale, CASE WHEN fatturatoTotale > 10000000 THEN 'Alto' WHEN fatturatoTotale > 1000000 THEN 'Medio' ELSE 'Basso' END AS categoriaFatturato",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è il primo e l'ultimo documento di vendita mai registrato per il cliente 'ACME SRL'?",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(doc:Documento) WHERE toLower(trim(c.name)) = 'acme srl' RETURN min(doc.dataEmissione) AS primoDocumento, max(doc.dataEmissione) AS ultimoDocumento",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Calcola la deviazione standard degli importi delle righe fattura per il ciclo attivo.",
        "output": "MATCH (:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN stDev(dr.importo) AS deviazioneStandardImporti",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Trova i prodotti in comune venduti all'azienda '1' e acquistati dal fornitore 'FORNITORE ESEMPIO SPA'.",
        "output": "MATCH (d:Ditta {dittaId: '1'})<-[:APPARTIENE_A]-(:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) WITH collect(DISTINCT a) AS prodottiVenduti MATCH (gf:GruppoFornitore)<-[:RAGGRUPPATO_SOTTO]-(:Fornitore)-[:HA_EMESSO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a2:Articolo) WHERE toLower(trim(gf.ragioneSociale)) = 'fornitore esempio spa' AND a2 IN prodottiVenduti RETURN a2.descrizione AS prodottoComune",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è il 95° percentile per valore delle righe documento di acquisto?",
        "output": "MATCH (:Fornitore)-[:HA_EMESSO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) RETURN percentileDisc(dr.importo, 0.95) AS percentile95Acquisti",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Per i primi 3 clienti per fatturato, mostra il loro prodotto più acquistato per valore.",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 WITH c, sum(dr.importo) AS fatturatoTotale ORDER BY fatturatoTotale DESC LIMIT 3 MATCH (c)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr2:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) WHERE dr2.tipoValore = 'Fatturato' AND dr2.importo > 0 WITH c, a, sum(dr2.importo) AS valoreProdotto ORDER BY c.name, valoreProdotto DESC WITH c, collect({prodotto: a.descrizione, valore: valoreProdotto})[0] AS prodottoTop RETURN c.name AS cliente, prodottoTop.prodotto AS prodottoPiuAcquistato, prodottoTop.valore",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Mostrami la crescita del fatturato mese su mese per l'anno 2024.",
        "output": "MATCH (:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 AND doc.dataEmissione.year = 2024 WITH doc.dataEmissione.month AS mese, sum(dr.importo) AS fatturatoMensile ORDER BY mese WITH collect(fatturatoMensile) AS fatturati RETURN [i in range(0, size(fatturati)-2) | (fatturati[i+1] - fatturati[i]) / fatturati[i] * 100] AS crescitaPercentualeMeseSuMese",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quali clienti hanno effettuato ordini ma non hanno ancora ricevuto una fattura?",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento {tipoValore: 'Ordinato'}) WITH DISTINCT c MATCH (c) WHERE NOT (c)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento {tipoValore: 'Fatturato'}) RETURN c.name AS clienteSenzaFattura",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Chi è il fornitore principale per ogni famiglia di prodotto, in base al valore acquistato?",
        "output": "MATCH (fam:Famiglia)<-[:INCLUSA_IN]-(:Sottofamiglia)<-[:APPARTIENE_A]-(:Articolo)<-[:RIGUARDA_ARTICOLO]-(dr:RigaDocumento)<-[:CONTIENE_RIGA]-(:Documento)<-[:HA_EMESSO]-(:Fornitore)-[:RAGGRUPPATO_SOTTO]->(gf:GruppoFornitore) WITH fam, gf, sum(dr.importo) AS valoreAcquistato ORDER BY fam.nome, valoreAcquistato DESC WITH fam, collect({fornitore: gf.ragioneSociale, valore: valoreAcquistato})[0] AS fornitoreTop RETURN fam.nome AS famiglia, fornitoreTop.fornitore AS fornitorePrincipale",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Elenca le province che non hanno né clienti né fornitori.",
        "output": "MATCH (l:Luogo) WHERE l.provincia IS NOT NULL WITH collect(DISTINCT l.provincia) AS tutteLeProvince MATCH (c:Cliente)-[:HAS_ADDRESS]->(lc:Luogo) WITH tutteLeProvince, collect(DISTINCT lc.provincia) AS provinceClienti MATCH (f:Fornitore)-[:SI_TROVA_A]->(lf:Luogo) WITH tutteLeProvince, provinceClienti, collect(DISTINCT lf.provincia) AS provinceFornitori CALL apoc.coll.subtract(tutteLeProvince, apoc.coll.union(provinceClienti, provinceFornitori)) YIELD value RETURN value AS provinciaVuota",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è il valore totale degli ordini in sospeso (ordinati ma non fatturati)?",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Ordinato' WITH c, doc, sum(dr.importo) AS valoreOrdinato WHERE NOT (doc)<-[:CONTIENE_RIGA]-(:RigaDocumento {tipoValore:'Fatturato'}) RETURN sum(valoreOrdinato) AS totaleOrdiniSospesi",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Mostrami i clienti che acquistano più di 10 articoli unici diversi.",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) WITH c, count(DISTINCT a) AS numeroArticoliUnici WHERE numeroArticoliUnici > 10 RETURN c.name AS cliente, numeroArticoliUnici ORDER BY numeroArticoliUnici DESC",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Per il prodotto più venduto per quantità, qual è il suo costo medio di acquisto?",
        "output": "MATCH (:Cliente)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(a:Articolo) WITH a, sum(dr.quantita) AS quantitaTotale ORDER BY quantitaTotale DESC LIMIT 1 MATCH (a)<-[:RIGUARDA_ARTICOLO]-(dr_acq:RigaDocumento)<-[:CONTIENE_RIGA]-(:Documento)<-[:HA_EMESSO]-(:Fornitore) RETURN a.descrizione AS prodotto, avg(dr_acq.importo / dr_acq.quantita) AS costoMedioUnitario",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Elenca i fornitori che trattano solo prodotti di una singola famiglia.",
        "output": "MATCH (gf:GruppoFornitore)<-[:RAGGRUPPATO_SOTTO]-(:Fornitore)-[:HA_EMESSO]->(:Documento)-[:CONTIENE_RIGA]->(:RigaDocumento)-[:RIGUARDA_ARTICOLO]->(:Articolo)-[:APPARTIENE_A]->(:Sottofamiglia)-[:INCLUSA_IN]->(fam:Famiglia) WITH gf, count(DISTINCT fam) AS numeroFamiglie WHERE numeroFamiglie = 1 RETURN gf.ragioneSociale AS fornitoreMonoFamiglia",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è la percentuale di fatturato generata dai clienti nella regione 'Umbria' sul totale?",
        "output": "MATCH (c:Cliente)-[:HAS_ADDRESS]->(l:Luogo) MATCH (c)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 WITH l.regione AS regione, sum(dr.importo) AS fatturatoRegione WITH collect({regione: regione, fatturato: fatturatoRegione}) AS fatturatiRegionali, sum(dr.importo) AS fatturatoTotale UNWIND fatturatiRegionali AS r WHERE toLower(trim(r.regione)) = 'umbria' RETURN r.fatturato AS fatturatoUmbria, fatturatoTotale, (r.fatturato / fatturatoTotale) * 100 AS percentuale",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Trova il cliente con il valore medio degli ordini più alto.",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 WITH c, doc, sum(dr.importo) AS valoreOrdine RETURN c.name AS cliente, avg(valoreOrdine) AS valoreMedioOrdine ORDER BY valoreMedioOrdine DESC LIMIT 1",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Mostrami i 3 mesi con il maggior numero di documenti di acquisto registrati.",
        "output": "MATCH (:Fornitore)-[:HA_EMESSO]->(doc:Documento) RETURN doc.dataEmissione.year AS anno, doc.dataEmissione.month AS mese, count(doc) AS numeroDocumenti ORDER BY numeroDocumenti DESC LIMIT 3",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Ci sono prodotti venduti a un prezzo inferiore al loro costo medio di acquisto?",
        "output": "MATCH (a:Articolo) OPTIONAL MATCH (a)<-[:RIGUARDA_ARTICOLO]-(dr_acq:RigaDocumento)<-[:CONTIENE_RIGA]-(:Documento)<-[:HA_EMESSO]-(:Fornitore) WITH a, avg(dr_acq.importo / dr_acq.quantita) AS costoMedio OPTIONAL MATCH (a)<-[:RIGUARDA_ARTICOLO]-(dr_ven:RigaDocumento)<-[:CONTIENE_RIGA]-(:Documento)<-[:HA_RICEVUTO]-(:Cliente) WHERE dr_ven.tipoValore = 'Fatturato' AND dr_ven.importo > 0 WITH a, costoMedio, avg(dr_ven.importo / dr_ven.quantita) AS prezzoMedioVendita WHERE prezzoMedioVendita < costoMedio RETURN a.descrizione AS prodottoInPerdita",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quale tipo di documento (codice) è più frequentemente associato al ciclo di acquisto?",
        "output": "MATCH (:Fornitore)-[:HA_EMESSO]->(doc:Documento)-[:IS_TYPE]->(dt:DocType) RETURN dt.codice AS tipoDocumento, count(doc) AS frequenza ORDER BY frequenza DESC LIMIT 1",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Elenca i fornitori situati in un luogo dove non abbiamo clienti.",
        "output": "MATCH (c:Cliente)-[:HAS_ADDRESS]->(l:Luogo) WITH collect(DISTINCT l.localita) AS localitaClienti MATCH (f:Fornitore)-[:SI_TROVA_A]->(lf:Luogo) WHERE NOT lf.localita IN localitaClienti MATCH (f)-[:RAGGRUPPATO_SOTTO]->(gf:GruppoFornitore) RETURN DISTINCT gf.ragioneSociale AS fornitoreIsolato",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quante note di credito clienti (NCC) abbiamo emesso in totale?",
        "output": "MATCH (doc:Documento)-[:IS_TYPE]->(dt:DocType {codice: 'NCC'}) RETURN count(doc) AS numeroNoteCredito",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Mostrami tutti gli ordini fornitori (OF) per maggio 2024.",
        "output": "MATCH (doc:Documento)-[:IS_TYPE]->(dt:DocType {codice: 'OF'}) WHERE doc.dataEmissione.year = 2024 AND doc.dataEmissione.month = 5 RETURN doc.documentoinstruction AS ordine",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è il valore totale delle fatture fornitori (FFA) ricevute?",
        "output": "MATCH (doc:Documento)-[:IS_TYPE]->(dt:DocType {codice: 'FFA'}) MATCH (doc)-[:CONTIENE_RIGA]->(dr:RigaDocumento) RETURN sum(dr.importo) AS totaleFattureFornitore",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Elenca i clienti a cui abbiamo inviato una nota di consegna clienti (DDTV) a Roma.",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:IS_TYPE]->(dt:DocType {codice: 'DDTV'}) WHERE toLower(trim(doc.tipoOriginale)) CONTAINS 'roma' RETURN DISTINCT c.name AS cliente",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è il valore fatturato presente solo nelle fatture clienti (FVC)?",
        "output": "MATCH (doc:Documento)-[:IS_TYPE]->(dt:DocType {codice: 'FVC'}) MATCH (doc)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' RETURN sum(dr.importo) AS totaleFatturatoInFVC",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Mostrami il totale ordinato attraverso gli ordini clienti (OC) per la ditta '1'.",
        "output": "MATCH (d:Ditta {dittaId: '1'})<-[:APPARTIENE_A]-(:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:IS_TYPE]->(dt:DocType {codice: 'OC'}) MATCH (doc)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Ordinato' RETURN sum(dr.importo) AS totaleOrdinatoDitta1",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Qual è il valore medio delle righe fatturate nelle note di consegna clienti (DDTV)?",
        "output": "MATCH (doc:Documento)-[:IS_TYPE]->(dt:DocType {codice: 'DDTV'}) MATCH (doc)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Bollato' RETURN avg(dr.importo) AS valoreMedioBollato",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Chi è il cliente con il valore fatturato più alto considerando solo le fatture immediate (FVC)?",
        "output": "MATCH (c:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:IS_TYPE]->(dt:DocType {codice: 'FVC'}) MATCH (doc)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN c.name AS cliente, sum(dr.importo) AS totale ORDER BY totale DESC LIMIT 1",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Per il cliente 'ACME SRL', confronta il totale ordinato e il totale fatturato.",
        "output": "MATCH (c:Cliente) WHERE toLower(trim(c.name)) = 'acme srl' OPTIONAL MATCH (c)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr_ord:RigaDocumento {tipoValore: 'Ordinato'}) WITH c, sum(dr_ord.importo) AS totaleOrdinato OPTIONAL MATCH (c)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr_fat:RigaDocumento {tipoValore: 'Fatturato'}) WHERE dr_fat.importo > 0 RETURN totaleOrdinato, sum(dr_fat.importo) AS totaleFatturato",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Elenca i documenti d'ordine clienti (OC) che non sono ancora stati completamente fatturati.",
        "output": "MATCH (doc:Documento)-[:IS_TYPE]->(dt:DocType {codice: 'OC'}) MATCH (doc)-[:CONTIENE_RIGA]->(dr:RigaDocumento {tipoValore: 'Ordinato'}) WITH doc, sum(dr.importo) AS valoreOrdinato MATCH (doc)-[:CONTIENE_RIGA]->(dr_fat:RigaDocumento {tipoValore: 'Fatturato'}) WITH doc, valoreOrdinato, sum(dr_fat.importo) AS valoreFatturato WHERE valoreOrdinato > valoreFatturato RETURN doc.documentoinstruction AS ordineNonSaldato",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Quale tipo di documento (nome) ha il valore medio più alto per riga (considerando solo i valori fatturati)?",
        "output": "MATCH (doc:Documento)-[:IS_TYPE]->(dt:DocType) MATCH (doc)-[:CONTIENE_RIGA]->(dr:RigaDocumento) WHERE dr.tipoValore = 'Fatturato' AND dr.importo > 0 RETURN dt.nome AS nomeTipoDocumento, avg(dr.importo) AS valoreMedioRiga ORDER BY valoreMedioRiga DESC LIMIT 1",
    },
    {
        "instruction": "Genera la query Cypher per rispondere a questo input in italiano",
        "input": "Mostrami il totale fatturato e il totale ordinato per ciascuna ditta.",
        "output": "MATCH (d:Ditta)<-[:APPARTIENE_A]-(c:Cliente) OPTIONAL MATCH (c)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr_ord:RigaDocumento {tipoValore: 'Ordinato'}) WITH d, c, sum(dr_ord.importo) AS totaleOrdinato OPTIONAL MATCH (c)-[:HA_RICEVUTO]->(:Documento)-[:CONTIENE_RIGA]->(dr_fat:RigaDocumento {tipoValore: 'Fatturato'}) WHERE dr_fat.importo > 0 WITH d, sum(totaleOrdinato) AS totaleOrdinatoDitta, sum(dr_fat.importo) AS totaleFatturatoDitta RETURN d.dittaId AS ditta, totaleOrdinatoDitta, totaleFatturatoDitta",
    },
]


# ==============================================================================
# == 3. LOGICA DI GENERAZIONE AUTOMATICA                                  ==
# ==============================================================================


def generate_variations(base_examples):
    """
    Analizza gli esempi di base e genera nuove varianti sostituendo i placeholder.
    """
    generated_data = []

    print("⚙️  Inizio generazione delle varianti...")

    for example in base_examples:
        original_input = example["input"]
        original_output = example["output"]

        # --- Genera varianti per CLIENTE ---
        if (
            "'barone di ferrante ezio'" in original_output.lower()
            or "'l\\'abbondanza srl'" in original_output.lower()
        ):
            for cliente in CLIENTI:
                new_input = original_input.replace(
                    "BARONE DI FERRANTE EZIO", cliente
                ).replace("L'ABBONDANZA SRL", cliente)
                new_output = original_output.replace(
                    "'barone di ferrante ezio'", f"'{cliente.lower()}'"
                ).replace("'l\\'abbondanza srl'", f"'{cliente.lower()}'")
                generated_data.append(
                    {
                        "instruction": example["instruction"],
                        "input": new_input,
                        "output": new_output,
                    }
                )

        # --- Genera varianti per TOP N ---
        if "limit 5" in original_output.lower() or "limit 3" in original_output.lower():
            for n in TOP_N_VALUES:
                new_input = re.sub(
                    r"\b[0-9]+\b", str(n), original_input
                )  # Sostituisce il primo numero che trova
                new_output = re.sub(
                    r"limit \d+", f"LIMIT {n}", original_output, flags=re.IGNORECASE
                )
                generated_data.append(
                    {
                        "instruction": example["instruction"],
                        "input": new_input,
                        "output": new_output,
                    }
                )

        # --- Genera varianti per ANNO ---
        if (
            "2023" in original_input
            or "2024" in original_input
            or "2025" in original_input
        ):
            for anno in ANNI:
                new_input = re.sub(r"\b202\d\b", anno, original_input)
                new_output = re.sub(r"\b202\d\b", anno, original_output)
                generated_data.append(
                    {
                        "instruction": example["instruction"],
                        "input": new_input,
                        "output": new_output,
                    }
                )

        # --- Genera varianti per FAMIGLIA ---
        if "'centrali frigo'" in original_output.lower():
            for famiglia in FAMIGLIE:
                new_input = original_input.replace("CENTRALI FRIGO", famiglia)
                new_output = original_output.replace(
                    "'centrali frigo'", f"'{famiglia.lower()}'"
                )
                generated_data.append(
                    {
                        "instruction": example["instruction"],
                        "input": new_input,
                        "output": new_output,
                    }
                )

        # --- Genera varianti per PRODOTTO ---
        if "'gas freon solstice n40 r448a'" in original_output.lower():
            for prodotto in PRODOTTI:
                new_input = original_input.replace(
                    "GAS FREON SOLSTICE N40 R448A", prodotto
                )
                new_output = original_output.replace(
                    "'gas freon solstice n40 r448a'", f"'{prodotto.lower()}'"
                )
                generated_data.append(
                    {
                        "instruction": example["instruction"],
                        "input": new_input,
                        "output": new_output,
                    }
                )

        # --- Genera varianti per LOCALITA/PROVINCIA ---
        if "'perugia'" in original_output.lower() or "'pg'" in original_output.lower():
            for i in range(len(LOCALITA)):
                # Variante per Località
                new_input_loc = original_input.replace("Perugia", LOCALITA[i])
                new_output_loc = original_output.replace(
                    "'perugia'", f"'{LOCALITA[i].lower()}'"
                )
                generated_data.append(
                    {
                        "instruction": example["instruction"],
                        "input": new_input_loc,
                        "output": new_output_loc,
                    }
                )
                # Variante per Provincia
                new_input_prov = original_input.replace("PG", PROVINCE[i])
                new_output_prov = original_output.replace(
                    "'pg'", f"'{PROVINCE[i].lower()}'"
                )
                generated_data.append(
                    {
                        "instruction": example["instruction"],
                        "input": new_input_prov,
                        "output": new_output_prov,
                    }
                )

        # =======================================================================
        # == NUOVO BLOCCO: Genera varianti per TIPO VALORE                    ==
        # =======================================================================
        if (
            "dr.tipovalore = 'fatturato'" in original_output.lower()
            and "sum(dr.importo)" in original_output.lower()
        ):
            for variant in TIPO_VALORE_VARIANTS:
                if variant["value"] == "Fatturato":
                    continue  # Salta l'originale

                # Sostituisci la parola chiave nella domanda
                new_input = original_input.replace(
                    "turnover", variant["keyword"]
                ).replace("fatturato", variant["keyword"])
                # Sostituisci il valore nella query
                new_output = original_output.replace(
                    "'Fatturato'", f"'{variant['value']}'"
                )

                generated_data.append(
                    {
                        "instruction": example["instruction"],
                        "input": new_input,
                        "output": new_output,
                    }
                )

        # =======================================================================
        # == NUOVO BLOCCO: Genera varianti per DOC TYPE                     ==
        # =======================================================================
        # Cerca un esempio che filtri per DocType e Fornitore
        if (
            "dt.codice = 'fna'" in original_output.lower()
            and "fornitore" in original_input.lower()
        ):
            for variant in DOCTYPE_VARIANTS:
                # Scegli un'entità casuale per la variante
                if variant["cycle"] == "CLIENTE":
                    entity_name = random.choice(CLIENTI)
                    new_input = f"Mostrami le {variant['user_term']} per il cliente '{entity_name}'"
                    new_output = (
                        f"MATCH (c:Cliente)-[:HA_RICEVUTO]->(doc:Documento)-[:IS_TYPE]->(dt:DocType) "
                        f"WHERE toLower(trim(c.name)) = '{entity_name.lower()}' AND dt.codice = '{variant['code']}' "
                        f"RETURN doc.documentoinstruction AS documento"
                    )
                else:  # FORNITORE
                    entity_name = random.choice(FORNITORI)
                    new_input = f"Mostrami le {variant['user_term']} per il fornitore '{entity_name}'"
                    new_output = (
                        f"MATCH (gf:GruppoFornitore)<-[:RAGGRUPPATO_SOTTO]-(:Fornitore)-[:HA_EMESSO]->(doc:Documento)-[:IS_TYPE]->(dt:DocType) "
                        f"WHERE toLower(trim(gf.ragioneSociale)) = '{entity_name.lower()}' AND dt.codice = '{variant['code']}' "
                        f"RETURN doc.documentoinstruction AS documento"
                    )

                generated_data.append(
                    {
                        "instruction": example["instruction"],
                        "input": new_input,
                        "output": new_output,
                    }
                )

    print(f"✅ Generati {len(generated_data)} nuovi esempi tramite sostituzione.")
    return generated_data


# ==============================================================================
# == 4. (AVANZATO) LOGICA DI PARAPHRASING CON LLM                          ==
# ==============================================================================
# NOTA: Questo richiede una chiamata a un LLM potente (es. gpt-4o-mini)


def paraphrase_inputs(examples_to_paraphrase, llm_client, num_variations=2):
    """
    Usa un LLM per riformulare le domande e aumentare la diversità linguistica.
    """
    paraphrased_data = []
    print(
        "\n⚙️  Inizio generazione delle parafrasi con LLM (potrebbe richiedere tempo)..."
    )

    paraphrase_prompt_template = """Your task is to rephrase a given user input in {num_variations} different ways, while keeping the original intent instructionentical. The input is in Italian. Respond with a JSON list of strings.

    Original input: "{input}"

    Respond ONLY with the JSON list.
    Example:
    Original input: "Qual è il fatturato totale del cliente ACME SRL?"
    JSON Response:
    [
        "Mostrami le vendite totali per ACME SRL.",
        "A quanto ammonta l'incasso generato da ACME SRL?",
        "Calcola il turnover di ACME SRL."
    ]
    """

    for i, example in enumerate(examples_to_paraphrase):
        print(f"  -> Parafrasando l'esempio {i+1}/{len(examples_to_paraphrase)}...")
        try:
            prompt = paraphrase_prompt_template.format(
                num_variations=num_variations, input=example["input"]
            )

            # --- QUI DOVRESTI FARE LA TUA CHIAMATA ALL'LLM ---
            # Esempio:
            # response = llm_client.invoke(prompt)
            # new_inputs = json.loads(response.content)

            # Per ora, simuliamo la risposta per non dipendere da una API key
            new_inputs = [
                f"Puoi calcolare {example['input'][9:]}",
                f"Dimmi {example['input'][9:]}",
            ]  # FINE SIMULAZIONE

            for new_q in new_inputs:
                paraphrased_data.append(
                    {
                        "instruction": example["instruction"],
                        "input": new_q,
                        "output": example["output"],
                    }
                )
        except Exception as e:
            print(f"    ⚠️  Errore durante la parafrasi: {e}")
            continue

    print(f"✅ Generati {len(paraphrased_data)} nuovi esempi tramite parafrasi.")
    return paraphrased_data


# ==============================================================================
# == 5. ESECUZIONE E SALVATAGGIO                                          ==
# ==============================================================================

if __name__ == "__main__":
    # --- Genera dati tramite sostituzione ---
    generated_variations = generate_variations(base_examples)

    # --- (Opzionale) Genera dati tramite parafrasi ---
    # Per attivarlo, devi avere un client LLM configurato.
    # Seleziona un sottoinsieme casuale di esempi da parafrasare per non esagerare
    # examples_for_paraphrasing = random.sample(base_examples + generated_variations, 20)
    # paraphrased_examples = paraphrase_inputs(examples_for_paraphrasing, llm_client=None) # Sostituisci None col tuo client

    # --- Combina e de-duplica ---
    # final_dataset = base_examples + generated_variations + paraphrased_examples
    final_dataset = base_examples + generated_variations

    # Rimuovi eventuali duplicati esatti
    unique_dataset = [dict(t) for t in {tuple(d.items()) for d in final_dataset}]

    print("\n" + "=" * 50)
    print(f"📊 STATISTICHE DATASET FINALE")
    print(f"  - Esempi di base: {len(base_examples)}")
    print(f"  - Esempi generati: {len(generated_variations)}")
    # print(f"  - Esempi parafrasati: {len(paraphrased_examples)}")
    print(f"  - Totale prima della de-duplicazione: {len(final_dataset)}")
    print(f"  - Totale finale (unici): {len(unique_dataset)}")
    print("=" * 50)

    # --- Salva il dataset in formato JSONL, instructioneale per il fine-tuning ---
    output_filename = "training_dataset.jsonl"
    with open(output_filename, "w", encoding="utf-8") as f:
        for entry in unique_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✅ Dataset salvato con successo in '{output_filename}'")
