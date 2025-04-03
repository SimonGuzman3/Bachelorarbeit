# Bachelorarbeit – Optimierte Auswahl von Heusteriken für das Knotenfärbungsproblem mit KI-Methoden

Dieses Repository enthält den vollständigen Code zur Bachelorarbeit von Simon Guzman. Ziel des Projekts ist die Evaluation und Kombination klassischer Färbungsheuristiken auf großen Graphinstanzen sowie die Anwendung von Machine Learning zur Auswahl des besten Algorithmus für einen gegebenen Graphen.

---

## Projektstruktur

```
├── ai_model_training.py         # Trainiert KI-Modelle zur Vorhersage der besten Heuristik
├── algorithmComparision.py      # Führt alle Heuristiken auf Graphen aus und speichert Ergebnisse
├── coloring_algorithms.py       # Implementierung von LF, DSatur, Welsh-Powell & Random-Greedy
├── generate_graph_instances.py  # Erzeugt neue Graph-Instanzen im DIMACS-Format
├── graph_result_processor.py    # Wandelt Resultate in ein ML-taugliches Format um
├── graph_tables/                # HDF5-Dateien mit Ergebnissen (LFS!)
├── graph_testinstances/         # Eingabedateien im DIMACS-Format (LFS!)
├── dataframe/                   # ML-Features & Labels
├── .gitattributes               # Git LFS Tracking für große Dateien
├── .gitignore                   # Ignorierte temporäre/irrelevante Dateien
```

---

## Verwendung

1. **Zusätzliche synthetische Graphen erzeugen:**
   ```bash
   python generate_graph_instances.py
   ```
   Dieser Schritt ist optional und nur notwendig, wenn mehr künstlich erzeugte Graphen zur Verfügung stehen sollen.

2. **Heuristiken ausführen & Ergebnisse speichern:**
   ⚠️ Hinweis: Nur notwendig, wenn **neue Graphen** hinzugefügt wurden, dann muss die Funktion `get_data()` in `algorithmComparision.py` bei **Zeile 215** auskommentiert werden.  
   Ansonsten **kommentiert lassen**, da sie sonst **alle Graphen neu färbt**, was sehr **zeitaufwendig** ist!

3. **ML-Daten vorbereiten und Modelle trainieren:**
   ```bash
   python ai_model_training.py
   ```
   In diesem Schritt werden die KI-Modelle trainiert und auf die Testinstanzen angewendet. Es ist einfach möglich neue KI-Modelle hinzufügen oder den Featurevekotr anzupassen.

---

## Heuristiken im Vergleich

- **LF (Largest First)**
- **DSatur**
- **Welsh-Powell**
- **Random Greedy (mehrfach wiederholt)**

Ziel ist es, die Anzahl der Farben und die Laufzeit zu minimieren. Die Modelle lernen aus strukturellen Graphfeatures wie Dichte, Knotengrad oder Clustering-Koeffizient.

---

## Genutzte ML-Modelle

- Random Forest  
- Gradient Boosting  
- Multi-Layer Perceptron (Neural Network)

> Die Modelle wählen automatisiert die beste Heuristik für unbekannte Graphen anhand gelernter Merkmale.

---

## Große Dateien

Die Dateien im Ordner `graph_tables/` und `graph_testinstances/` werden über **[Git LFS](https://git-lfs.github.com/)** verwaltet.

> ❗ Bei einem `git clone` installiere Git LFS zuerst mit:
>
> ```bash
> git lfs install
> git lfs pull
> ```

---

## Lizenz

Dieses Repository ist nur zu wissenschaftlichen und privaten Zwecken nutzbar. Eine kommerzielle oder modifizierte Weiterverwendung ist ohne Zustimmung nicht erlaubt.

---

## Autor

- **Simon Guzman**
- Kontakt: [GitHub-Profil](https://github.com/SimonGuzman3)

---

## Wenn dir das Projekt gefällt

...freue ich mich über ein ⭐ auf [GitHub](https://github.com/SimonGuzman3/Bachelorarbeit)
