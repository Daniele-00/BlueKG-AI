def _get_memory_resolution(
    session,
    term_key: str,
    preferred_labels: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Recupera l'entità dalla memoria, MA:
      - prima prova a trovare una chiave 'simile' (fuzzy),
      - poi applica le preferred_labels per l'intent.
    """
    mem_store = getattr(session, "resolved_entities", None)
    if not mem_store:
        return None

    # 1) Trova eventuale chiave alias simile
    effective_key = term_key
    alias_key = _find_best_memory_key(mem_store, term_key)
    if alias_key is not None:
        effective_key = alias_key

    entry = mem_store.get(effective_key)
    if not entry:
        return None

    # Compat vecchio formato: {"name": "...", "label": "..."}
    if isinstance(entry, dict) and "name" in entry and "label" in entry:
        entry = {entry["label"]: entry}

    # entry: { "Cliente": {...}, "GruppoFornitore": {...}, ... }

    if not preferred_labels:
        # Nessuna preferenza di intent -> prendi il primo ruolo disponibile
        return next(iter(entry.values()))

    # Con preferenze: cerca il primo ruolo compatibile
    for label in preferred_labels:
        if label in entry:
            return entry[label]

    # Memoria esiste ma solo con ruoli non compatibili
    logger.info(
        "[Resolver P1 Memory] '%s' presente solo con ruoli %s, "
        "non compatibili con preferenze %s. Non la uso.",
        effective_key,
        list(entry.keys()),
        preferred_labels,
    )
    return None
