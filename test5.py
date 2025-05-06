import numpy as np

def reduceMatrix(matrix):
    """
    Redukuje macierz poprzez odjęcie najmniejszych wartości w każdej kolumnie,
    a następnie w każdym wierszu. Zwraca zredukowaną macierz oraz sumę odjętych wartości.
    """
    minInCols = np.min(matrix, axis=0)
    newMatrix = matrix - minInCols
    reduced = np.sum(minInCols)

    minInRows = np.min(newMatrix, axis=1)
    return newMatrix - minInRows[:, np.newaxis], reduced + np.sum(minInRows)

def zwieksz_liczbe_zer(macierz, wiersze_wykreslone, kolumny_wykreslone):
    """
    Modyfikuje macierz kosztów zgodnie z krokiem algorytmu węgierskiego,
    aby umożliwić znalezienie większej liczby zer niezależnych w kolejnej iteracji.
    """

    nieprzykryte = ~wiersze_wykreslone[:, np.newaxis] & ~kolumny_wykreslone[np.newaxis, :]
    przykryte_dwiema_liniami = wiersze_wykreslone[:, np.newaxis] & kolumny_wykreslone[np.newaxis, :]
    print(f'\nWykreślone linie:\n{przykryte_dwiema_liniami}\n')

    min_val = np.min(macierz[nieprzykryte]) if np.any(nieprzykryte) else 0
    print(f'Wartość wyliczona do zmodyfikowania macierzy = {min_val}')

    nowa_macierz = macierz.copy()
    nowa_macierz[nieprzykryte] -= min_val
    nowa_macierz[przykryte_dwiema_liniami] += min_val

    return nowa_macierz

def wyswietl_macierz_z_zerami(macierz, zera_niezalezne):
    """
    Tworzy macierz z oznaczeniami:
    0 - dowolne liczby (nie będące zerami w macierzy wejściowej)
    1 - zera niezależne (z listy zera_niezalezne)
    2 - zera zależne (pozostałe zera z macierzy wejściowej)

    Args:
        macierz (np.ndarray): Macierz wejściowa (po redukcji/modyfikacji).
        zera_niezalezne (list[tuple[int, int]] | None): Lista współrzędnych
                                                        zer niezależnych lub None.

    Returns:
        np.ndarray: Macierz z oznaczeniami 0, 1, 2.
    """
    n, m = macierz.shape
    oznaczenia = np.zeros((n, m), dtype=int)

    oznaczenia[macierz == 0] = 2

    if zera_niezalezne:
        for i, j in zera_niezalezne:
            if 0 <= i < n and 0 <= j < m:
                oznaczenia[i, j] = 1
            else:
                print(f"[OSTRZEŻENIE] Współrzędne zera niezależnego ({i}, {j}) poza zakresem macierzy ({n}x{m}).")

    return oznaczenia



def wyznaczanie_zer_niezaleznych(matrix):
    """
    Wyznacza maksymalny możliwy zbiór zer niezależnych (skojarzenie)
    w macierzy kosztów za pomocą algorytmu zachłannego. Preferuje
    wiersze/kolumny z mniejszą liczbą dostępnych zer.

    Args:
        matrix (np.ndarray): Macierz wejściowa (zazwyczaj po redukcji).

    Returns:
        list[tuple[int, int]]: Lista krotek (wiersz, kolumna)
                               reprezentujących znalezione zera niezależne.
                               Pusta lista, jeśli nie ma zer lub nie można znaleźć.
    """
    rows, cols = matrix.shape
    zera_niezalezne = []
    wiersze_zajete = np.zeros(rows, dtype=bool)         #Jednowymiarowe wektory nie dwuwymiarowe maceirze
    kolumny_zajete = np.zeros(cols, dtype=bool)

    while True:
        best_zero_pos = None
        min_zeros_in_line = float('inf')
        found_in_row = True # To jak dla mnie bez sensu tu

        # Szukaj w wierszach
        for r in range(rows):
            if not wiersze_zajete[r]:
                count = 0
                candidate_pos = None
                for c in range(cols):
                    if not kolumny_zajete[c] and matrix[r, c] == 0:
                        count += 1
                        if candidate_pos is None: candidate_pos = (r, c)
                if 0 < count < min_zeros_in_line:
                    min_zeros_in_line = count
                    best_zero_pos = candidate_pos
                    found_in_row = True
                    if count == 1: break # Najlepszy możliwy

        # Szukaj w kolumnach (jeśli nie było jednoznacznego w wierszu)
        if not (best_zero_pos and min_zeros_in_line == 1 and found_in_row):
            for c in range(cols):
                 if not kolumny_zajete[c]:
                    count = 0
                    candidate_pos = None
                    for r in range(rows):
                         if not wiersze_zajete[r] and matrix[r, c] == 0:
                             count += 1
                             if candidate_pos is None: candidate_pos = (r, c)
                    if 0 < count < min_zeros_in_line:
                        min_zeros_in_line = count
                        best_zero_pos = candidate_pos
                        found_in_row = False   # to  nie jest używane już więc też nie ma sensu tego modyfikować
                        if count == 1: break # Najlepszy możliwy

        # Dodaj znalezione zero lub zakończ
        if best_zero_pos is not None:
            r_idx, c_idx = best_zero_pos
            zera_niezalezne.append((r_idx, c_idx))
            wiersze_zajete[r_idx] = True
            kolumny_zajete[c_idx] = True
            print(f"  -> Znaleziono i dodano zero niezależne: ({r_idx}, {c_idx})")
        else:
            print("  -> Nie można znaleźć więcej zer niezależnych.")
            break # Koniec pętli while

    return zera_niezalezne


def pokryj_zera_min_liczba_linii(macierz, zera_niezalezne):
    """
    Znajduje minimalny zestaw linii pokrywających wszystkie zera w macierzy.
    Zwraca maski zaznaczonych wierszy i kolumn.

    Używa metody bazującej na klasycznym opisie algorytmu węgierskiego:
    1. Zaznacz wiersze bez zer niezależnych.
    2. Iteruj: zaznacz kolumny z zerami w zaznaczonych wierszach,
              zaznacz wiersze z zerami niezależnymi w zaznaczonych kolumnach.
    3. Linie rysujemy przez niezaznaczone wiersze i zaznaczone kolumny.
    """
    n = macierz.shape[0]
    zaznaczone_wiersze = np.ones(n, dtype=bool)
    zaznaczone_kolumny = np.zeros(n, dtype=bool)

    # Wiersze bez zer niezależnych
    for r in range(n):
        if any(z[0] == r for z in zera_niezalezne):
            zaznaczone_wiersze[r] = False

    zmiana = True
    while zmiana:
        zmiana = False
        # Kolumny z zerami w zaznaczonych wierszach
        for r in range(n):
            if zaznaczone_wiersze[r]:
                for c in range(n):
                    if macierz[r, c] == 0 and not zaznaczone_kolumny[c]:
                        zaznaczone_kolumny[c] = True
                        zmiana = True
        # Wiersze z niezależnymi zerami w zaznaczonych kolumnach
        for c in range(n):
            if zaznaczone_kolumny[c]:
                for r, c2 in zera_niezalezne:
                    if c2 == c and not zaznaczone_wiersze[r]:
                        zaznaczone_wiersze[r] = True
                        zmiana = True

    # Linie rysujemy przez NIEzaznaczone wiersze i zaznaczone kolumny
    linie_wiersze = ~zaznaczone_wiersze
    linie_kolumny = zaznaczone_kolumny

    return linie_wiersze, linie_kolumny





def algorytm_wegierski(macierz):
    """
    Pełna implementacja algorytmu węgierskiego dla kwadratowej macierzy kosztów.
    """
    print("=== ALGORYTM WĘGIERSKI ===")
    print("\nMacierz początkowa:\n", macierz)
    matrix = macierz.copy()

    matrix, suma_redukcji = reduceMatrix(matrix)
    print("\nPo redukcji:\n", matrix)
    print(F"\nSuma redukcji = {suma_redukcji}\n")

    count = 1
    while True:
        print(f"\n== {count} ITERACJA ALGORYTMU ==\n")

        zera_niezalezne = wyznaczanie_zer_niezaleznych(matrix)
        print(f"Znaleziono {len(zera_niezalezne)} zer niezależnych.")
        print(f"\nZnalezione zera niezależne: {zera_niezalezne}")
        print("\nMacierz z oznaczeniami zer:")
        oznaczenia_po_kroku2 = wyswietl_macierz_z_zerami(matrix, zera_niezalezne)
        print(oznaczenia_po_kroku2)

        if len(zera_niezalezne) == matrix.shape[0]:
            print("\nPełne przyporządkowanie znalezione.")
            print("\n--- WYNIK KOŃCOWY ---")
            print("\nZera niezależne (przypisania):", zera_niezalezne)
            print("Wartości tych zer (przypisania):", [int(macierz[i, j]) for i, j in zera_niezalezne])
            koszt_calkowity = sum(macierz[i, j] for i, j in zera_niezalezne)
            print("Minimalny koszt przypisania:", koszt_calkowity)         #koszt całkowity = suma redukcji !!!!

            return zera_niezalezne, suma_redukcji

        wiersze, kolumny = pokryj_zera_min_liczba_linii(matrix, zera_niezalezne)
        matrix = zwieksz_liczbe_zer(matrix, wiersze, kolumny)
        print("\nMacierz po modyfikacji:\n")
        print(matrix)

        count += 1




print("--- START PRZYKŁADU ---")

# Dane wejściowe
m = [[12, 14, 17, 9, 23, 21], 
     [15, 10, 12, 18, 16, 14], 
     [8, 13, 15, 17, 10, 22],
     [18, 11, 14, 13, 16, 12], 
     [22, 16, 13, 21, 9, 15], 
     [19, 15, 11, 20, 18, 10]]

M = np.asarray(m)
algorytm_wegierski(M)

print("\n--- KONIEC PRZYKŁADU ---")
