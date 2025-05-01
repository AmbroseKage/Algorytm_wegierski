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

    min_val = np.min(macierz[nieprzykryte]) if np.any(nieprzykryte) else 0

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

# --- Wadliwa funkcja (zakomentowana) ---

    # def wyznaczanie_zer(matrix):
    # """
    # Wyznacza maskę zer w macierzy kosztów.
    # """
    # mask = np.zeros(6, int)
    # for i in range(len(matrix)):
    #     for j in range(len(matrix)):
    #         if matrix[i][j] == 0:
    #             mask[i][j] = 1
    # flag = 0
    # for i in range(len(matrix)):
    #     if 1 in matrix[:, i]:
    #         flag += 1
    # if flag == len(matrix):
    #     return matrix
    # pass

# --- Poprawna funkcja wyznaczania zer niezależnych ---
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
    print("[wyznaczanie_zer_niezaleznych - POPRAWNA] Uruchomiono.")
    rows, cols = matrix.shape
    zera_niezalezne = []
    wiersze_zajete = np.zeros(rows, dtype=bool)
    kolumny_zajete = np.zeros(cols, dtype=bool)

    while True:
        best_zero_pos = None
        min_zeros_in_line = float('inf')
        found_in_row = True

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
                        found_in_row = False
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


print("--- START PRZYKŁADU ---")

# Dane wejściowe
m = [[12, 14, 17, 9, 23, 21], 
     [15, 10, 12, 18, 16, 14], 
     [8, 13, 15, 17, 10, 22],
     [18, 11, 14, 13, 16, 12], 
     [22, 16, 13, 21, 9, 15], 
     [19, 15, 11, 20, 18, 10]]

M = np.asarray(m)
print("Macierz początkowa:\n", M)

biezaca_macierz, suma_redukcji = reduceMatrix(M)
print("\n--- Po kroku 1: Redukcja ---")
print("Zredukowana macierz:\n", biezaca_macierz)
print("Suma redukcji:", suma_redukcji)


print("\n--- Po kroku 2: Wyznaczanie Zer Niezależnych ---")
znalezione_zera = wyznaczanie_zer_niezaleznych(biezaca_macierz)
print(f"\nZnalezione zera niezależne: {znalezione_zera}")
print(f"Liczba znalezionych zer: {len(znalezione_zera)}")

print("\nMacierz z oznaczeniami (po kroku 2):")
oznaczenia_po_kroku2 = wyswietl_macierz_z_zerami(biezaca_macierz, znalezione_zera)
print(oznaczenia_po_kroku2)

print("\n--- KONIEC PRZYKŁADU ---")
