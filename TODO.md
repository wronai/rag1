# TODO

## Wysoki priorytet
- [ ] Obsługa parametrów chunkowania z poziomu CLI (`--chunk-size`, `--chunk-overlap`).
- [ ] Cache FAISS na dysku, aby uniknąć ponownej wektoryzacji przy kolejnych uruchomieniach na tym samym zbiorze dokumentów.

## Średni priorytet
- [ ] Testy integracyjne pokrywające pełny przepływ: ładowanie dokumentów → retrieval → generacja.
- [ ] Konfiguracja CI uruchamiająca testy oraz sanity check dla `rag.py`.

## Niski priorytet
- [ ] Przykładowy notebook pokazujący eksperymenty z różnymi embedderami.
- [ ] Rozszerzenie dokumentacji o sekcję "Najczęstsze błędy" wraz z rozwiązaniami.
