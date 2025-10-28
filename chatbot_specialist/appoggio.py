// --- LEGENDA AUTOMATICA ---
      const uniqueLabels = Array.from(new Set(nodes.flatMap(n => n.labels || ["Node"])));
      const legend = d3.select("#{element_id}-legend");

      // MODIFICA 1: Aggiunto font-size al titolo
      legend.html("<strong style='color:#38bdf8; font-size: 1.1rem;'>Legenda nodi:</strong><br/>" +
        uniqueLabels.map(lbl => {{
          const col = color(lbl);
          
          // MODIFICA 2: Aumentato font-size e margine dell'intero elemento
          return `<span style='display:inline-flex;align-items:center;margin-right:12px; font-size: 1rem;'>
                    
                    <span style='width:15px;height:15px;background:${{col}};
                    border-radius:50%;display:inline-block;margin-right:8px;border:1px solid #334155;'></span>
                    
                    ${{lbl}}
                  </span>`;
        }}).join(" ")); // Aggiungi .join(" ") per unire gli elementi dell'array in una stringa