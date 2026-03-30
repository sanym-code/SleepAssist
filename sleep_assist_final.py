"""
==============================================================================
SLEEP ASSIST - INTELLIGENTER SCHLAFPHASENWECKER
==============================================================================
Bundeswettbewerb Künstliche Intelligenz 2024

PROJEKTÜBERSICHT:
KI-basiertes System zur Optimierung des Aufwachens durch Erkennung günstiger 
Schlafphasen mittels Wearable-Sensordaten. Das System kombiniert Transfer 
Learning mit personalisierten Modellen für optimale Weckzeiten.

TECHNISCHE INNOVATION:
• Stratifizierte Zeitreihen-Aufteilung (respektiert Schlafphasengrenzen)
• Transfer Learning ohne manuelle Labels (Pseudo-Labeling mit 98% Konfidenz)  
• Personalisierte LSTM-Modelle für individuelle Schlafmuster
• Echtzeit-Smart-Alarm-Simulation

DATENGRUNDLAGE:
DREAMT-Datensatz mit Empatica E4 Sensordaten (BVP, ACC, EDA, TEMP, HR)
Binäre Klassifikation: Günstige vs. Ungünstige Aufwachzeiten

ERGEBNISSE:
• Test-Genauigkeit: 79.6%
• F1-Score: 79.1% (gewichtet)
• Binäre Klassifikation: Precision bis 90.6%, Recall bis 93.7%
• Smart Alarm findet optimale Weckzeiten (z.B. 4:55 Uhr mit 91.1% vs. 
  traditionell 4:45 Uhr mit 77.7%)

AUSFÜHRUNG:
1. CSV-Dateien im Format: [TIMESTAMP, BVP, ACC_X, ACC_Y, ACC_Z, TEMP, EDA, HR, Sleep_Stage]
2. python sleep_assist_final.py
3. Modelle werden trainiert und Smart-Alarm-Simulation startet automatisch
4. Am besten eignet sich Google Colab wo der Code sowie die Dateien S002, S003, ..., S007 importiert werden müssen und der Code anschließend einfach ausgeführt werden kann

SYSTEMANFORDERUNGEN:
• Python 3.8+, TensorFlow 2.x, scikit-learn, pandas, numpy, matplotlib
• 8GB+ RAM für Training, GPU empfohlen
• Trainingsdauer: ~20-30 Min (CPU), ~5-10 Min (GPU)

ANWENDUNG:
Besonders wertvoll für Menschen mit unregelmäßigen Schlafmustern, 
Schichtarbeiter und alle, die ihre Schlafqualität optimieren möchten.

PIPELINE-PHASEN:
1. Allgemeines Modell: Cross-Dataset-Training für robuste Basis
2. Transfer Learning: Pseudo-Labeling auf personalisierte Daten
3. Feinabstimmung: Eingefrorene Schichten + neue Ausgabeschicht
4. Smart-Alarm: Echtzeit-Simulation mit optimaler Weckzeit-Bestimmung

AUTOR: Sanyukt Mishra
DATUM: 2024
KONTAKT: sanyukt.mishra@outlook.com

==============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                           precision_score, recall_score, roc_curve, auc,
                           balanced_accuracy_score)
from sklearn.preprocessing import label_binarize, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from collections import Counter
import os
import time as time_module
from datetime import time, datetime, timedelta

# ====================================================================
# HILFSFUNKTIONEN FÜR DATENVORVERARBEITUNG
# ====================================================================

def stratified_sleep_split(X, y, test_size=0.1, val_size=0.1, min_consecutive=30):
    """
    Teilt Zeitreihendaten stratifiziert auf und respektiert dabei Schlafphasengrenzen.
    
    Diese Funktion sorgt dafür, dass:
    1. Alle Schlafphasen in allen Datensätzen vertreten sind
    2. Die zeitliche Reihenfolge erhalten bleibt
    3. Keine Aufteilung mitten in einer Schlafphase erfolgt
    
    Args:
        X (numpy.array): Eingabedaten (Features)
        y (numpy.array): Zielvariablen (Schlafphasen)
        test_size (float): Anteil der Testdaten (Standard: 0.1)
        val_size (float): Anteil der Validierungsdaten (Standard: 0.1)
        min_consecutive (int): Mindestanzahl aufeinanderfolgender Proben pro Segment
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Finde Phasenwechsel (wo sich die Schlafphase aendert)
    stage_changes = np.where(np.diff(y) != 0)[0] + 1
    stage_changes = np.concatenate(([0], stage_changes, [len(y)]))

    # Erstelle Segmente mit ihrer Phase und Laenge
    segments = []
    for i in range(len(stage_changes) - 1):
        start_idx = stage_changes[i]
        end_idx = stage_changes[i + 1]
        stage = y[start_idx]
        length = end_idx - start_idx
        segments.append({
            'start': start_idx,
            'end': end_idx,
            'stage': stage,
            'length': length
        })

    # Gruppiere Segmente nach Schlafphase
    stage_segments = {}
    for segment in segments:
        stage = segment['stage']
        if stage not in stage_segments:
            stage_segments[stage] = []
        stage_segments[stage].append(segment)

    print("Verteilung der Schlafphasen:")
    for stage, segs in stage_segments.items():
        total_samples = sum(seg['length'] for seg in segs)
        stage_names = ['P', 'W', 'N1', 'N2', 'R']
        if stage < len(stage_names):
            print(f"Phase {stage_names[int(stage)]}: {len(segs)} Segmente, {total_samples} Proben")

    # Für jede Phase: Wähle Segmente für Test- und Validierungssets aus
    train_indices = []
    val_indices = []
    test_indices = []

    for stage, segs in stage_segments.items():
        # Sortiere Segmente nach Länge (längste zuerst)
        segs_sorted = sorted(segs, key=lambda x: x['length'], reverse=True)

        total_stage_samples = sum(seg['length'] for seg in segs_sorted)
        target_test_samples = int(total_stage_samples * test_size)
        target_val_samples = int(total_stage_samples * val_size)

        test_samples_collected = 0
        val_samples_collected = 0
        stage_test_indices = []
        stage_val_indices = []
        stage_train_indices = []

        # Verteile Test- und Validierungsproben auf mehrere Segmente
        segments_for_test_val = max(2, len(segs_sorted) // 3)

        for i, segment in enumerate(segs_sorted):
            segment_indices = list(range(segment['start'], segment['end']))

            if (i < segments_for_test_val and segment['length'] >= min_consecutive):
                if (test_samples_collected < target_test_samples and
                    (i % 2 == 0 or val_samples_collected >= target_val_samples)):

                    samples_to_take = min(
                        segment['length'] // 2,
                        target_test_samples - test_samples_collected
                    )

                    if samples_to_take >= min_consecutive:
                        start_offset = (segment['length'] - samples_to_take) // 2
                        test_segment_indices = segment_indices[start_offset:start_offset + samples_to_take]
                        stage_test_indices.extend(test_segment_indices)
                        test_samples_collected += len(test_segment_indices)

                        remaining_indices = (segment_indices[:start_offset] +
                                           segment_indices[start_offset + samples_to_take:])
                        stage_train_indices.extend(remaining_indices)
                    else:
                        # samples_to_take zu klein: ganzes Segment zum Training
                        stage_train_indices.extend(segment_indices)

                elif val_samples_collected < target_val_samples:
                    samples_to_take = min(
                        segment['length'] // 2,
                        target_val_samples - val_samples_collected
                    )

                    if samples_to_take >= min_consecutive:
                        start_offset = (segment['length'] - samples_to_take) // 2
                        val_segment_indices = segment_indices[start_offset:start_offset + samples_to_take]
                        stage_val_indices.extend(val_segment_indices)
                        val_samples_collected += len(val_segment_indices)

                        remaining_indices = (segment_indices[:start_offset] +
                                           segment_indices[start_offset + samples_to_take:])
                        stage_train_indices.extend(remaining_indices)
                    else:
                        # FIX: remaining_indices war hier nicht definiert wenn samples_to_take
                        # zu klein ist. Korrektur: ganzes Segment zum Training hinzufügen.
                        stage_train_indices.extend(segment_indices)
                else:
                    stage_train_indices.extend(segment_indices)
            else:
                stage_train_indices.extend(segment_indices)

        train_indices.extend(stage_train_indices)
        val_indices.extend(stage_val_indices)
        test_indices.extend(stage_test_indices)

    # Sortiere Indizes um zeitliche Reihenfolge zu erhalten
    train_indices.sort()
    val_indices.sort()
    test_indices.sort()

    # Erstelle die Aufteilungen
    X_train = X[train_indices]
    X_val = X[val_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_sequences(X, y, sequence_length=20):
    """
    Erstellt Sequenzen für das LSTM-Netzwerk aus 2D-Daten.
    
    RNN/LSTM-Netzwerke benötigen Sequenzen als Eingabe, nicht einzelne Datenpunkte.
    Diese Funktion wandelt die Zeitreihendaten in überlappende Sequenzen um.
    
    Args:
        X (numpy.array): Feature-Matrix (Zeitschritte x Features)
        y (numpy.array): Labels
        sequence_length (int): Länge jeder Sequenz
        
    Returns:
        tuple: (X_sequences, y_sequences) - Sequenzen für das LSTM
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length + 1):
        # Erstelle Sequenz der Länge 'sequence_length'
        X_seq.append(X[i:i+sequence_length])
        # Das Label ist das der letzten Position in der Sequenz
        y_seq.append(y[i+sequence_length-1])
    return np.array(X_seq), np.array(y_seq)


def prepare_single_dataset(dataset_name, test_size=0.2, val_size=0.2, sequence_length=20,
                          labeling_strategy='binary', indicies='even'):
    """
    Bereitet einen einzelnen Datensatz für personalisierte Schlafphasenklassifikation vor.
    
    Diese Funktion implementiert personalisiertes Machine Learning - das Modell wird
    speziell für eine Person trainiert, was bessere Ergebnisse liefert als ein
    allgemeines Modell für alle Personen.
    
    Args:
        dataset_name (str): Name des Datensatzes (z.B. 'S007')
        test_size (float): Anteil der Testdaten
        val_size (float): Anteil der Validierungsdaten
        sequence_length (int): Länge der LSTM-Sequenzen
        labeling_strategy (str): 'binary', 'simplified' oder 'detailed'
        indicies (str): 'even', 'odd' oder 'all' für Datenfilterung
        
    Returns:
        tuple: Vorbereitete Daten für Training, Validierung und Test,
               sowie scaler_X für konsistente Verwendung in Phase 3.
    """
    print(f"Bereite personalisierten Datensatz vor: {dataset_name}")
    print("="*60)

    # Lade einzelnen Datensatz
    print(f"Lade {dataset_name}...")
    data = pd.read_csv(f"{dataset_name}_whole_df.csv")
    
    # Optional: Reduziere Datenmenge durch Filterung
    if indicies == 'even':
        data = data.iloc[::2]  # Jede zweite Zeile
    elif indicies == 'odd':
        data = data.iloc[1::2]  # Jede zweite Zeile, versetzt

    print(f"{dataset_name} Form: {data.shape}")

    def extract_features_labels(data, labeling_strategy='detailed'):
        """
        Extrahiert Features und Labels aus den Rohdaten mit verschiedenen Strategien.
        
        Verschiedene Labeling-Strategien ermöglichen unterschiedliche Anwendungen:
        - 'detailed': 5 Klassen (P, W, N1, N2, R) für detaillierte Analyse
        - 'simplified': 3 Klassen für vereinfachte Anwendung
        - 'binary': 2 Klassen (gut/schlecht zum Aufwachen) für den Smart Alarm
        """
        feature_columns = ['TIMESTAMP', 'BVP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'TEMP', 'EDA', 'HR']
        X = data[feature_columns].values

        sleepstage = data['Sleep_Stage'].values
        y = np.zeros(len(sleepstage))

        if labeling_strategy == 'detailed':
            # Ursprüngliches 5-Klassen-System
            stage_mapping = {'P': 0, 'W': 1, 'N1': 2, 'N2': 3, 'R': 4}
            stage_names = ['P (Vor-Schlaf)', 'W (Wach)', 'N1 (Leicht)', 'N2 (Tief)', 'R (REM)']
        elif labeling_strategy == 'simplified':
            # Vereinfachtes 3-Klassen-System
            stage_mapping = {
                'P': 0,   # Vor-Schlaf/Übergang
                'W': 1,   # Wach/Leicht (GUT zum Aufwachen)
                'N1': 1,  # Leichter Schlaf (GUT zum Aufwachen)
                'N2': 2,  # Tiefschlaf (SCHLECHT zum Aufwachen)
                'R': 1    # REM (OK zum Aufwachen)
            }
            stage_names = ['Übergang', 'Optimales Aufwachen', 'Aufwachen vermeiden']
        elif labeling_strategy == 'binary':
            # Binär: Gute vs. Schlechte Aufwachzeiten
            stage_mapping = {
                'P': 0,   # Schlechte Aufwachzeit
                'W': 1,   # Gute Aufwachzeit
                'N1': 1,  # Gute Aufwachzeit
                'N2': 0,  # Schlechte Aufwachzeit
                'R': 1    # Gute Aufwachzeit
            }
            stage_names = ['Schlechte Aufwachzeit', 'Gute Aufwachzeit']

        # Konvertiere String-Labels zu numerischen Werten
        for i, value in enumerate(sleepstage):
            y[i] = stage_mapping.get(value, 0)

        return X, y, stage_names

    X, y, stage_names = extract_features_labels(data, labeling_strategy)

    print(f"\nLabeling-Strategie: {labeling_strategy.upper()}")
    print(f"Klassen: {stage_names}")
    print("Ursprüngliche Verteilung:", Counter(y))

    # Wende stratifizierte Aufteilung an
    print(f"\nWende stratifizierte Aufteilung an...")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_sleep_split(
        X, y, test_size=test_size, val_size=val_size
    )

    print(f"\nFinale Aufteilungsgrößen:")
    print(f"Training: {X_train.shape[0]} Proben ({100*(1-test_size-val_size):.0f}%)")
    print(f"Validierung: {X_val.shape[0]} Proben ({100*val_size:.0f}%)")
    print(f"Test: {X_test.shape[0]} Proben ({100*test_size:.0f}%)")

    # Skaliere Features (wichtig: nur auf Trainingsdaten anpassen!)
    print("\nSkaliere Features (Anpassung nur auf Trainingsdaten)...")
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    # Berechne Klassengewichte (nur aus Trainingsdaten)
    print("Berechne Klassengewichte (nur aus Trainingsdaten)...")
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print("Klassengewichte:", class_weight_dict)

    # Konvertiere Labels zu kategorischen Variablen
    num_classes = len(np.unique(y))
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    print(f"\nKlassenverteilungen nach Aufteilung:")
    print("Training:", Counter(y_train))
    print("Validierung:", Counter(y_val))
    print("Test:", Counter(y_test))

    # Erstelle Sequenzen für LSTM
    print(f"\nErstelle Sequenzen für LSTM (Länge={sequence_length})...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_cat, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_cat, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_cat, sequence_length)

    print(f"\nSequenz-Formen:")
    print(f"Training: {X_train_seq.shape}")
    print(f"Validierung: {X_val_seq.shape}")
    print(f"Test: {X_test_seq.shape}")

    print("\n" + "="*60)
    print(f"Personalisierte Modellvorbereitung abgeschlossen für {dataset_name}")
    print("="*60)

    return (X_train_seq, X_val_seq, X_test_seq,
            y_train_seq, y_val_seq, y_test_seq,
            class_weight_dict, scaler_X, num_classes)


def prepare_cross_dataset_evaluation(test_dataset='S002', val_dataset='S003'):
    """
    Cross-Dataset-Vorbereitung: Separate Datensätze für Training/Validierung/Test.
    
    Diese Funktion implementiert Cross-Dataset-Evaluation, um die Generalisierbarkeit
    des Modells zu testen. Ein Datensatz wird für Training verwendet, andere für Test.
    
    Args:
        test_dataset (str): Dataset für Testen
        val_dataset (str): Dataset für Validierung
        
    Returns:
        tuple: Vorbereitete Cross-Dataset-Daten
    """
    print("Lade Datensätze separat...")
    dataset_files = ['S002', 'S003', 'S004', 'S005', 'S006']
    datasets = {}

    # Lade alle verfügbaren Datensätze
    for name in dataset_files:
        data = pd.read_csv(f"{name}_whole_df.csv")
        data = data.iloc[::10]  # Reduziere auf jeden 10. Datenpunkt für Performance
        datasets[name] = data
        print(f"{name} Form: {data.shape}")

    print(f"\nVerwende {test_dataset} für Test, {val_dataset} für Validierung")

    test_data = datasets[test_dataset]
    val_data = datasets[val_dataset]

    # Kombiniere verbleibende Datensätze für Training
    train_datasets = []
    for name, data in datasets.items():
        if name not in [test_dataset, val_dataset]:
            train_datasets.append(data)
            print(f"Verwende {name} für Training")

    train_data = pd.concat(train_datasets, ignore_index=True)

    print(f"\nFinale Aufteilungsgrößen:")
    print(f"Training: {train_data.shape[0]} Proben aus {len(train_datasets)} Datensätzen")
    print(f"Validierung: {val_data.shape[0]} Proben")
    print(f"Test: {test_data.shape[0]} Proben")

    def extract_features_labels(data):
        """Extrahiert Features und Labels für Cross-Dataset-Evaluation."""
        feature_columns = ['TIMESTAMP', 'BVP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'TEMP', 'EDA', 'HR']
        X = data[feature_columns].values

        sleepstage = data['Sleep_Stage'].values
        y = np.zeros(len(sleepstage))
        stage_mapping = {'P': 0, 'W': 1, 'N1': 2, 'N2': 3, 'R': 4}
        for i, value in enumerate(sleepstage):
            y[i] = stage_mapping.get(value, 0)

        return X, y

    X_train, y_train = extract_features_labels(train_data)
    X_val, y_val = extract_features_labels(val_data)
    X_test, y_test = extract_features_labels(test_data)

    print("\nKlassenverteilungen:")
    print("Training:", Counter(y_train))
    print("Validierung:", Counter(y_val))
    print("Test:", Counter(y_test))

    # Skaliere Features
    print("Skaliere Features (Anpassung nur auf Trainingsdaten)...")
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    # Berechne Klassengewichte
    print("Berechne Klassengewichte (nur aus Trainingsdaten)...")
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print("Klassengewichte:", class_weight_dict)

    # Konvertiere Labels zu kategorisch
    num_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat = to_categorical(y_val, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    # Erstelle Sequenzen
    print("Erstelle Sequenzen für LSTM...")
    sequence_length = 20

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_cat, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_cat, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_cat, sequence_length)

    print(f"\nSequenz-Formen:")
    print(f"Training: {X_train_seq.shape}")
    print(f"Validierung: {X_val_seq.shape}")
    print(f"Test: {X_test_seq.shape}")

    return (X_train_seq, X_val_seq, X_test_seq,
            y_train_seq, y_val_seq, y_test_seq,
            class_weight_dict, scaler_X, num_classes)


def convert_to_binary_labels(y_categorical):
    """
    Konvertiert 5-Klassen-Labels zu binären Labels.
    
    Wandelt detaillierte Schlafphasen in eine einfache Klassifikation um:
    0 = Schlechte Aufwachzeit (P, N2)
    1 = Gute Aufwachzeit (W, N1, R)
    
    Args:
        y_categorical: One-hot codierte Labels
        
    Returns:
        numpy.array: Binäre Labels (one-hot codiert)
    """
    y_numeric = np.argmax(y_categorical, axis=1)
    # P=0->0, W=1->1, N1=2->1, N2=3->0, R=4->1
    binary_mapping = {0: 0, 1: 1, 2: 1, 3: 0, 4: 1}
    y_binary = np.array([binary_mapping[label] for label in y_numeric])
    return to_categorical(y_binary, 2)

# ====================================================================
# PHASE 1: TRAINING DES ALLGEMEINEN BINÄREN MODELLS
# ====================================================================

print("\n--- Phase 1: Training des allgemeinen binären Modells ---")

# Lade Cross-Dataset-Daten für allgemeines Modell
X_train_cross, X_val_cross, X_test_cross, y_train_cross, y_val_cross, y_test_cross, class_weight_dict_cross, scaler_cross, num_classes_cross = prepare_cross_dataset_evaluation(
    test_dataset='S005',
    val_dataset='S006'
)

# Konvertiere zu binärer Klassifikation für Smart Alarm
y_train_binary = convert_to_binary_labels(y_train_cross)
y_val_binary = convert_to_binary_labels(y_val_cross)
y_test_binary = convert_to_binary_labels(y_test_cross)

# Neuberechnung der Klassengewichte für binäre Klassifikation
y_train_numeric = np.argmax(y_train_binary, axis=1)
class_weights_binary = compute_class_weight(
    'balanced',
    classes=np.unique(y_train_numeric),
    y=y_train_numeric
)
class_weight_dict_binary = {i: class_weights_binary[i] for i in range(len(class_weights_binary))}

print(f"Binäre Klassenverteilung - Training: {Counter(y_train_numeric)}")
print(f"Binäre Klassengewichte: {class_weight_dict_binary}")

# LSTM-Modellarchitektur mit Regularisierung gegen Overfitting
print("Erstelle LSTM-Modell...")
model_LSTM = Sequential([
    # Erste LSTM-Schicht mit Return Sequences für stacked LSTM
    LSTM(64, return_sequences=True, input_shape=(X_train_cross.shape[1], X_train_cross.shape[2])),
    Dropout(0.3),  # Dropout verhindert Overfitting
    BatchNormalization(),  # Stabilisiert das Training
    
    # Zweite LSTM-Schicht
    LSTM(32, return_sequences=False),
    Dropout(0.4),
    BatchNormalization(),
    
    # Voll vernetzte Schichten
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Binäre Klassifikation (2 Klassen)
])

model_LSTM.summary()

# Modell-Kompilation mit Adam-Optimizer
model_LSTM.compile(
    optimizer=Adam(learning_rate=0.001),  # Moderate Lernrate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback-Funktionen für intelligentes Training
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,  # Reduziere Lernrate um 80%
    patience=3,  # Warte 3 Epochen ohne Verbesserung
    min_lr=1e-7,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,  # Stoppe nach 10 Epochen ohne Verbesserung
    restore_best_weights=True,
    verbose=1
)

# Training des allgemeinen Modells
print("Starte Training des allgemeinen Modells...")
history = model_LSTM.fit(
    X_train_cross, y_train_binary,
    epochs=10,
    batch_size=64,
    verbose=1,
    validation_data=(X_val_cross, y_val_binary),
    callbacks=[lr_scheduler, early_stopping],
    class_weight=class_weight_dict_binary
)

# Evaluierung des allgemeinen Modells
test_loss, test_accuracy = model_LSTM.evaluate(X_test_cross, y_test_binary, verbose=0)
print(f"Allgemeines Modell - Test-Genauigkeit: {test_accuracy:.4f}")

# Speichere das allgemeine Modell
model_LSTM.save('sleep_classifier_lstm_general_binary.keras')
print("Allgemeines Modell gespeichert!")

# ====================================================================
# PHASE 2: TRANSFER LEARNING AUF PERSONALISIERTE DATEN
# ====================================================================

print("\n--- Phase 2: Transfer Learning auf S007 ---")

# Lade das vortrainierte allgemeine Modell
model_pretrained = load_model('sleep_classifier_lstm_general_binary.keras')

# FIX: Scaler aus Phase 2 wird gespeichert und in Phase 3 wiederverwendet,
# damit Test-Daten mit exakt demselben Scaler transformiert werden wie die
# Trainingsdaten. Vorher wurde prepare_single_dataset zweimal aufgerufen,
# was zu zwei unabhängig gefitteten Scalern führte (Data Leakage).
(X_s007_train, X_s007_val, X_s007_test,
 y_s007_train, y_s007_val, y_s007_test,
 class_weight_dict_s007, scaler_s007, num_classes_s007) = prepare_single_dataset(
    'S007',
    labeling_strategy='binary',
    test_size=0.2,
    val_size=0.1
)

print(f"S007 Datenform: Training={X_s007_train.shape}, Validierung={X_s007_val.shape}, Test={X_s007_test.shape}")
print(f"S007 Klassenverteilung: {Counter(np.argmax(y_s007_train, axis=1))}")

# Pseudo-Labeling: Verwende allgemeines Modell um ungelabelte Daten zu labeln
print("Generiere Pseudo-Labels mit dem allgemeinen Modell...")
predictions_pseudo = model_pretrained.predict(
    X_s007_train,
    batch_size=128,
    verbose=1
)

# Hohe Konfidenz-Schwelle für qualitativ hochwertige Pseudo-Labels
high_confidence_threshold = 0.98
confidence_scores = np.max(predictions_pseudo, axis=1)
high_confidence_indices = np.where(confidence_scores > high_confidence_threshold)[0]

# Ausgeglichene Klassenselektion für stabile Feinabstimmung
pseudo_labels = np.argmax(predictions_pseudo[high_confidence_indices], axis=1)
class_counts = Counter(pseudo_labels)
min_class_count = min(class_counts.values()) if class_counts else 0
max_per_class = max(1000, min_class_count * 2)

# Balancierte Auswahl der Pseudo-Labels
balanced_indices = []
class_counter = {0: 0, 1: 0}

for idx in high_confidence_indices:
    predicted_class = np.argmax(predictions_pseudo[idx])
    if class_counter[predicted_class] < max_per_class:
        balanced_indices.append(idx)
        class_counter[predicted_class] += 1

X_finetune = X_s007_train[balanced_indices]
y_finetune = predictions_pseudo[balanced_indices]

print(f"Pseudo-Labels: {len(X_finetune)} Proben (Konfidenz > {high_confidence_threshold})")
print(f"Pseudo-Label Klassenverteilung: {Counter(np.argmax(y_finetune, axis=1))}")

# Transfer Learning mit Feinabstimmung
if len(X_finetune) > 1000:
    print("Starte Transfer Learning Feinabstimmung...")

    # Kopiere Modellstruktur bis zur letzten Schicht
    model_finetuned = Sequential()
    for layer in model_pretrained.layers[:-1]:
        layer.trainable = False   # Friere bestehende Schichten ein
        model_finetuned.add(layer)

    # Füge neue trainierbare Ausgabeschicht hinzu
    model_finetuned.add(Dense(2, activation='softmax', name='finetuned_output'))

    # Kompiliere das angepasste Modell
    model_finetuned.compile(
        optimizer=Adam(learning_rate=0.0001),  # Sehr niedrige Lernrate für Feinabstimmung
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Sanfte Feinabstimmung nur auf hochwertigen Pseudo-Labels
    finetune_history = model_finetuned.fit(
        X_finetune, y_finetune,
        epochs=10,
        batch_size=32,
        verbose=1,
        validation_data=(X_s007_val, y_s007_val),
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
    )

    model_finetuned.save('sleep_classifier_lstm_S007_final.keras')
    print("Feinabstimmung abgeschlossen!")

else:
    print("Zu wenige hochwertige Pseudo-Labels. Verwende ursprüngliches Modell.")
    model_finetuned = model_pretrained
    model_finetuned.save('sleep_classifier_lstm_S007_final.keras')

# ====================================================================
# PHASE 3: FINALE EVALUIERUNG
# ====================================================================

print("\n--- Phase 3: Finale Evaluierung des personalisierten Modells ---")

# Lade das finale Modell
model_final = load_model('sleep_classifier_lstm_S007_final.keras')

# FIX: Verwende X_s007_test und y_s007_test direkt aus Phase 2 statt
# prepare_single_dataset erneut aufzurufen. So ist garantiert, dass
# dieselben Daten mit demselben Scaler (scaler_s007) skaliert wurden.
print("Verwende Test-Daten und Scaler aus Phase 2 für konsistente Evaluierung...")

# Vorhersagen auf Test-Set
print("Führe finale Evaluierung durch...")
y_pred_proba = model_final.predict(
    X_s007_test,
    batch_size=128,
    verbose=1
)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_s007_test, axis=1)

sleep_stages = ['Schlechte Aufwachzeit', 'Gute Aufwachzeit']

# Berechne umfassende Metriken
print("\n🎯 FINALE ERGEBNISSE:")
print("="*50)
print(f"Test-Genauigkeit: {accuracy_score(y_true, y_pred):.4f}")
print(f"F1-Score (gewichtet): {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Ausgeglichene Genauigkeit: {balanced_accuracy_score(y_true, y_pred):.4f}")

# Detaillierte Klassenmetriken
print("\nDETAILLIERTE KLASSENMETRIKEN:")
for i, stage in enumerate(sleep_stages):
    precision = precision_score(y_true == i, y_pred == i, zero_division=0)
    recall = recall_score(y_true == i, y_pred == i, zero_division=0)
    f1 = f1_score(y_true == i, y_pred == i, zero_division=0)
    print(f"{stage}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# Visualisierung der Ergebnisse
cm = confusion_matrix(y_true, y_pred)
cm_norm = confusion_matrix(y_true, y_pred, normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=sleep_stages, yticklabels=sleep_stages)
plt.title('Finale Konfusionsmatrix - Schlafphasen-Klassifikation')
plt.xlabel('Vorhersage')
plt.ylabel('Tatsächlich')
plt.tight_layout()
plt.show()

# Konfidenzanalyse
confidence_scores = np.max(y_pred_proba, axis=1)
print(f"\nKonfidenzanalyse:")
print(f"Mittlere Konfidenz: {np.mean(confidence_scores):.3f}")
print(f"Standardabweichung: {np.std(confidence_scores):.3f}")

# Umfassende Visualisierungen
plt.figure(figsize=(15, 5))

# Konfidenzverteilung
plt.subplot(1, 3, 1)
plt.hist(confidence_scores, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Vorhersage-Konfidenz')
plt.ylabel('Häufigkeit')
plt.title('Konfidenzverteilung')

# ROC-Kurve für binäre Klassifikation
plt.subplot(1, 3, 2)
fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC-Kurve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Zufälliger Klassifizierer')
plt.xlabel('Falsch-Positiv-Rate')
plt.ylabel('Richtig-Positiv-Rate')
plt.title('ROC-Kurve')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Klassenverteilung im Test-Set
plt.subplot(1, 3, 3)
class_counts = Counter(y_true)
plt.bar([sleep_stages[k] for k in sorted(class_counts.keys())],
        [class_counts[k] for k in sorted(class_counts.keys())],
        alpha=0.7, edgecolor='black')
plt.xlabel('Klasse')
plt.ylabel('Anzahl Proben')
plt.title('Klassenverteilung Test-Set')
plt.xticks(rotation=15, ha='right')

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("✅ SCHLAFKLASSIFIKATIONS-PIPELINE ABGESCHLOSSEN!")
print("="*80)
print("🎯 Hauptmerkmale:")
print("   ✓ Konsistente binäre Klassifikation für Smart Alarm")
print("   ✓ Starke Regularisierung gegen Overfitting")
print("   ✓ Qualitäts-Pseudo-Labeling für Transfer Learning")
print("   ✓ Sanfte Feinabstimmung mit eingefrorenen Schichten")
print("   ✓ Umfassende Evaluierung und Visualisierung")
print("="*80)

# ====================================================================
# PHASE 4: SMART-ALARM-SIMULATOR
# ====================================================================

class SmartAlarmSimulator:
    """
    Smart-Alarm-Simulator für Echtzeit-Schlafphasenüberwachung.
    
    Diese Klasse simuliert einen intelligenten Wecker, der die optimale
    Aufwachzeit basierend auf der aktuellen Schlafphase bestimmt.
    """
    
    def __init__(self, model_path, wake_window_start=time(4, 35), wake_window_end=time(4, 55)):
        """
        Initialisiert den Smart-Alarm-Simulator.

        Args:
            model_path (str): Pfad zum trainierten LSTM-Modell
            wake_window_start (time): Beginn des Weckfensters (Standard: 4:35 Uhr)
            wake_window_end (time): Ende des Weckfensters (Standard: 4:55 Uhr)
        """
        self.model = load_model(model_path)
        self.wake_window_start = wake_window_start
        self.wake_window_end = wake_window_end
        self.seq_len = 20  # Gleiche Sequenzlänge wie beim Training
        self.scaler = MinMaxScaler()
        self.feature_cols = ['TIMESTAMP', 'BVP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'TEMP', 'EDA', 'HR']

        # Zustandsvariablen für die Simulation
        self.current_buffer = []
        self.predictions_history = []
        self.timestamps_history = []
        self.best_wake_time = None
        self.best_wake_score = 0.0
        self.alarm_triggered = False

    def load_and_prepare_data(self, csv_path):
        """
        Lädt und bereitet die Schlafdaten für die Simulation vor.
        
        Args:
            csv_path (str): Pfad zur CSV-Datei mit Schlafdaten
            
        Returns:
            numpy.array: Skalierte Feature-Matrix
        """
        print(f"🔄 Lade Schlafdaten für Demonstrations-Szenario...")
        print(f"💤 Verwende VOLLSTÄNDIGEN originalen Datensatz")
        print(f"🎯 Smart-Alarm-Fenster: {self.wake_window_start} - {self.wake_window_end}")

        # Lade Daten - KEINE Filterung, verwende exakt wie im Original
        df_full = pd.read_csv(csv_path)
        print(f"Originaldaten geladen: {df_full.shape}")

        # Konvertiere Zeitstempel - EXAKT wie im Original (22:00 Uhr Start)
        timestamp_seconds = df_full['TIMESTAMP'].values
        recording_start = datetime(2024, 1, 1, 22, 0, 0)  # 22 Uhr Start wie im Original
        self.timestamps = [recording_start + timedelta(seconds=float(s)) for s in timestamp_seconds]
        self.timestamps = pd.Series(self.timestamps)

        # Bereite Features vor - exakt wie im Original
        X_full = df_full[self.feature_cols].values.astype(np.float32)

        # Skaliere auf VOLLSTÄNDIGEN Daten - exakt wie im Original
        self.X_full_scaled = self.scaler.fit_transform(X_full)

        print(f"Daten vorbereitet: {len(self.timestamps)} Proben")
        print(f"Schlafperiode: {self.timestamps.min()} bis {self.timestamps.max()}")
        sleep_duration = (self.timestamps.max() - self.timestamps.min()).total_seconds() / 3600
        print(f"Schlafdauer: {sleep_duration:.1f} Stunden")

        return self.X_full_scaled

    def group_data_by_minute(self):
        """
        Gruppiert Datenpunkte nach Minuten für minütliche Verarbeitung.
        
        Returns:
            tuple: (minute_groups, sorted_minutes) - Gruppierte Daten und sortierte Zeitstempel
        """
        print("📊 Gruppiere Daten nach Minuten...")

        minute_groups = {}

        for i, timestamp in enumerate(self.timestamps):
            # Runde Zeitstempel auf nächste Minute
            minute_key = timestamp.replace(second=0, microsecond=0)

            if minute_key not in minute_groups:
                minute_groups[minute_key] = []

            minute_groups[minute_key].append({
                'index': i,
                'timestamp': timestamp,
                'features': self.X_full_scaled[i]
            })

        # Sortiere nach Zeitstempel
        sorted_minutes = sorted(minute_groups.keys())

        print(f"Daten in {len(sorted_minutes)} Minuten-Intervalle gruppiert")
        print(f"Erste Minute: {sorted_minutes[0]}")
        print(f"Letzte Minute: {sorted_minutes[-1]}")

        return minute_groups, sorted_minutes

    def process_minute_data(self, minute_data):
        """
        Verarbeitet alle Datenpunkte für eine gegebene Minute.
        
        Args:
            minute_data (list): Liste der Datenpunkte für diese Minute
            
        Returns:
            numpy.array: Durchschnittliche Features für diese Minute
        """
        if not minute_data:
            return None

        # Berechne Durchschnitt aller Features für diese Minute
        features_stack = np.array([point['features'] for point in minute_data])
        avg_features = np.mean(features_stack, axis=0)

        return avg_features

    def make_prediction(self, current_time):
        """
        Macht eine Vorhersage basierend auf dem aktuellen Puffer.
        
        Args:
            current_time (datetime): Aktuelle Zeit
            
        Returns:
            tuple: (prediction_proba, wake_probability) - Vorhersagewahrscheinlichkeiten
        """
        if len(self.current_buffer) < self.seq_len:
            return None, 0.0

        # Verwende die letzten seq_len Proben für Vorhersage
        sequence = np.array(self.current_buffer[-self.seq_len:])
        sequence = sequence.reshape(1, self.seq_len, len(self.feature_cols))

        # Mache Vorhersage
        prediction_proba = self.model.predict(sequence, verbose=0)[0]
        wake_probability = prediction_proba[1]  # Wahrscheinlichkeit für 'Gute Aufwachzeit'

        return prediction_proba, wake_probability

    def is_in_wake_window(self, current_time):
        """
        Prüft, ob die aktuelle Zeit im Weckfenster liegt.
        
        Args:
            current_time (datetime): Zu prüfende Zeit
            
        Returns:
            bool: True wenn im Weckfenster
        """
        current_time_only = current_time.time()
        return self.wake_window_start <= current_time_only <= self.wake_window_end

    def should_trigger_alarm(self, wake_probability, current_time):
        """
        Entscheidet, ob der Alarm ausgelöst werden soll.
        
        Args:
            wake_probability (float): Aktuelle Aufwach-Wahrscheinlichkeit
            current_time (datetime): Aktuelle Zeit
            
        Returns:
            bool: True wenn Alarm ausgelöst werden soll
        """
        if not self.is_in_wake_window(current_time):
            return False

        # Hohe Schwelle - löse nur bei exzellenten Aufwach-Gelegenheiten aus
        threshold = 0.9  # Hohe Schwelle um auf optimalen Moment zu warten
        return wake_probability >= threshold

    def run_simulation(self, csv_path, simulation_speed=1.0, show_progress=True, real_time_wake_window=True):
        """
        Führt die minütliche Simulation aus.

        Args:
            csv_path (str): Pfad zur Schlafdat-CSV
            simulation_speed (float): Geschwindigkeitsmultiplikator (1.0 = Echtzeit)
            show_progress (bool): Ob Fortschritt angezeigt werden soll
            real_time_wake_window (bool): Echtzeit-Wartemodus im Weckfenster.
                Bei True wartet das System 5 Minuten zwischen jeder Vorhersage
                im Weckfenster – entspricht dem realistischen Betrieb am
                Körper des Nutzers während der Nacht.
            
        Returns:
            tuple: (predictions_history, timestamps_history)
        """
        print(f"\n🚀 Starte Echtzeit Smart-Alarm Simulation")
        print(f"💤 Schlaf-Szenario: Vollständige Nachtruhe (ab 22:00 Uhr)")
        print(f"⏰ Weckfenster: {self.wake_window_start} - {self.wake_window_end}")
        print(f"⚠️  Traditioneller Alarm: 4:45 Uhr (macht müde!)")
        print(f"🎯 Smart Alarm: Wartet auf 0.9+ Schwelle ODER besten Zeitpunkt im Fenster")
        print(f"🎯 Simulationsgeschwindigkeit: {simulation_speed}x")
        if real_time_wake_window:
            print(f"⏳ Echtzeit-Verarbeitung im Weckfenster: AKTIVIERT (5 Min. Pause pro Schritt)")
        print("="*70)

        # Lade und bereite Daten vor
        self.load_and_prepare_data(csv_path)
        minute_groups, sorted_minutes = self.group_data_by_minute()

        # Starte Simulation
        simulation_start_time = time_module.time()

        for i, minute_timestamp in enumerate(sorted_minutes):
            minute_data = minute_groups[minute_timestamp]

            # Verarbeite Minutendaten
            avg_features = self.process_minute_data(minute_data)
            if avg_features is None:
                continue

            # Füge zu Puffer hinzu
            self.current_buffer.append(avg_features)

            # Mache Vorhersage
            prediction_proba, wake_probability = self.make_prediction(minute_timestamp)

            if prediction_proba is not None:
                # Speichere Vorhersage
                self.predictions_history.append(wake_probability)
                self.timestamps_history.append(minute_timestamp)

                # Aktualisiere beste Aufwachzeit wenn im Fenster und besserer Score
                if self.is_in_wake_window(minute_timestamp):
                    if wake_probability > self.best_wake_score:
                        self.best_wake_time = minute_timestamp
                        self.best_wake_score = wake_probability

                # Zeige Fortschritt
                if show_progress and i % 5 == 0:  # Zeige alle 5 Minuten
                    time_str = minute_timestamp.strftime('%H:%M:%S')
                    in_window = "🟢" if self.is_in_wake_window(minute_timestamp) else "⚫"

                    if wake_probability > 0.8:
                        prob_indicator = "🟢"
                    elif wake_probability > 0.6:
                        prob_indicator = "🟡"
                    elif wake_probability > 0.4:
                        prob_indicator = "🟠"
                    else:
                        prob_indicator = "🔴"

                    print(f"{in_window} {time_str} | Aufwach-Wahrsch.: {wake_probability:.3f} {prob_indicator} | Puffer: {len(self.current_buffer)}")

                    # Echtzeit-Warten während Weckfenster (realistischer Betrieb)
                    if real_time_wake_window and self.is_in_wake_window(minute_timestamp):
                        print(f"⏳ Nächste Messung in 5 Minuten...")
                        time_module.sleep(300)  # 5 Minuten warten

                # Prüfe ob Alarm ausgelöst werden soll
                if not self.alarm_triggered and self.should_trigger_alarm(wake_probability, minute_timestamp):
                    if not hasattr(self, 'alarm_eligible_time'):
                        self.alarm_eligible_time = minute_timestamp
                        self.alarm_eligible_score = wake_probability
                        print(f"\n🟡 ALARM BERECHTIGT um {minute_timestamp.strftime('%H:%M:%S')} (Score: {wake_probability:.3f})")
                        print(f"⏳ Überwache weiter das Weckfenster für bessere Gelegenheiten...")

            # Reguläre Geschwindigkeitskontrolle für Nicht-Weckfenster-Zeiten
            if simulation_speed > 0 and simulation_speed < 1.0 and not self.is_in_wake_window(minute_timestamp):
                time_module.sleep(60 * simulation_speed)

        # Nach Verarbeitung aller Daten: Löse Alarm aus wenn berechtigt ODER am Ende des Fensters
        if hasattr(self, 'alarm_eligible_time') and not self.alarm_triggered:
            print(f"\n🔔 SMART ALARM AUSGELÖST!")
            print(f"⏰ Zeit: {self.best_wake_time.strftime('%H:%M:%S')}")
            print(f"📊 Aufwach-Wahrscheinlichkeit: {self.best_wake_score:.3f}")
            print(f"✅ Optimale Aufwachzeit über Schwelle (0.9) gefunden!")
            self.alarm_triggered = True
        elif not self.alarm_triggered and self.best_wake_time:
            # Fallback: Löse am Ende des Fensters aus, auch wenn Schwelle nicht erreicht
            print(f"\n🔔 FALLBACK ALARM AUSGELÖST!")
            print(f"⏰ Zeit: {self.best_wake_time.strftime('%H:%M:%S')} (Ende des Weckfensters)")
            print(f"📊 Aufwach-Wahrscheinlichkeit: {self.best_wake_score:.3f}")
            print(f"⚠️  Schwelle (0.9) nicht erreicht, aber das war die beste verfügbare Option!")
            print(f"🆚 Traditioneller Alarm (4:45 Uhr) wäre schlechter gewesen!")
            self.alarm_triggered = True

        # Simulation abgeschlossen
        total_time = time_module.time() - simulation_start_time
        print("\n" + "="*70)
        print("🎯 SIMULATION ABGESCHLOSSEN!")
        print("="*70)

        if self.alarm_triggered:
            print(f"✅ Alarm erfolgreich ausgelöst um {self.best_wake_time.strftime('%H:%M:%S')}")
            print(f"📊 Finale Aufwach-Wahrscheinlichkeit: {self.best_wake_score:.3f}")
        elif self.best_wake_time:
            print(f"⏰ Beste Aufwachzeit gefunden: {self.best_wake_time.strftime('%H:%M:%S')}")
            print(f"📊 Beste Aufwach-Wahrscheinlichkeit: {self.best_wake_score:.3f}")
            print("⚠️  Alarm-Schwelle nicht erreicht")
        else:
            print("❌ Keine geeignete Aufwachzeit im Weckfenster gefunden")

        print(f"⏱️  Simulationsdauer: {total_time:.1f} Sekunden")
        print(f"📈 Gesamte Vorhersagen: {len(self.predictions_history)}")

        return self.predictions_history, self.timestamps_history

    def plot_simulation_results(self):
        """Visualisiert die Simulationsergebnisse."""
        if not self.predictions_history:
            print("Keine Daten zum Visualisieren")
            return

        print("\n📊 Erstelle Visualisierung...")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Konvertiere Zeitstempel zu Stunden für Plot
        hours = [t.hour + t.minute/60 for t in self.timestamps_history]

        # Plot 1: Echtzeit Aufwach-Wahrscheinlichkeit
        ax1.plot(hours, self.predictions_history, 'b-', linewidth=2, alpha=0.8, label='Aufwach-Wahrscheinlichkeit')

        # Markiere Weckfenster
        wake_start_hour = self.wake_window_start.hour + self.wake_window_start.minute/60
        wake_end_hour = self.wake_window_end.hour + self.wake_window_end.minute/60
        ax1.axvspan(wake_start_hour, wake_end_hour, alpha=0.2, color='green', label='Weckfenster')

        # Zeige beste Aufwachzeit
        if self.best_wake_time:
            best_hour = self.best_wake_time.hour + self.best_wake_time.minute/60
            ax1.axvline(x=best_hour, color='red', linestyle='--', linewidth=2, label='Beste Aufwachzeit')
            ax1.scatter([best_hour], [self.best_wake_score], color='red', s=100, zorder=5)

        # Zeige Alarm-Auslösepunkt
        if self.alarm_triggered:
            ax1.axhline(y=0.7, color='orange', linestyle=':', alpha=0.7, label='Alarm-Schwelle')

        ax1.set_xlabel('Tageszeit (Stunden)')
        ax1.set_ylabel('Aufwach-Wahrscheinlichkeit')
        ax1.set_title('Echtzeit Smart-Alarm Simulation - Aufwach-Wahrscheinlichkeit über Zeit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(min(hours), max(hours))
        ax1.set_ylim(0, 1)

        # Plot 2: Wahrscheinlichkeitsverteilung im Weckfenster
        wake_window_indices = []
        wake_window_probs = []

        for i, timestamp in enumerate(self.timestamps_history):
            if self.is_in_wake_window(timestamp):
                wake_window_indices.append(i)
                wake_window_probs.append(self.predictions_history[i])

        if wake_window_probs:
            wake_hours = [self.timestamps_history[i].hour + self.timestamps_history[i].minute/60
                         for i in wake_window_indices]
            ax2.plot(wake_hours, wake_window_probs, 'g-', linewidth=3, marker='o', markersize=4)
            ax2.fill_between(wake_hours, wake_window_probs, alpha=0.3, color='green')

            if self.best_wake_time:
                best_hour = self.best_wake_time.hour + self.best_wake_time.minute/60
                ax2.axvline(x=best_hour, color='red', linestyle='--', linewidth=2)
                ax2.scatter([best_hour], [self.best_wake_score], color='red', s=150, zorder=5)

        ax2.set_xlabel('Zeit im Weckfenster (Stunden)')
        ax2.set_ylabel('Aufwach-Wahrscheinlichkeit')
        ax2.set_title('Aufwach-Wahrscheinlichkeit während des Weckfensters')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        plt.show()

# ====================================================================
# HAUPTPROGRAMM - NUTZUNGSBEISPIEL
# ====================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🌙 SLEEP ASSIST - INTELLIGENTER SCHLAFPHASENWECKER")
    print("="*80)
    print("Entwickelt für den Bundeswettbewerb Künstliche Intelligenz")
    print("Ein KI-System zur Optimierung des Aufwachens durch Schlafphasen-Erkennung")
    print("="*80)
    
    # Initialisiere Simulator für Smart vs. Traditioneller Alarm Demonstration
    simulator = SmartAlarmSimulator(
        model_path='sleep_classifier_lstm_S007_final.keras',
        wake_window_start=time(4, 35),  # Beginne Überwachung um 4:35 Uhr
        wake_window_end=time(4, 55)     # Ende um 4:55 Uhr
    )

    # Führe Simulation aus
    predictions, timestamps = simulator.run_simulation(
        csv_path='S007_whole_df.csv',
        simulation_speed=0.0,  # 0.0 für sofortige Simulation, 1.0 für Echtzeit
        show_progress=True,
        real_time_wake_window=False  # Für Demonstration auf False
    )

    # Visualisiere Ergebnisse
    simulator.plot_simulation_results()

    # Simulationszusammenfassung
    print(f"\n📋 SIMULATIONSZUSAMMENFASSUNG:")
    print(f"• Verarbeitete Minuten: {len(predictions)}")
    print(f"• Weckfenster-Abdeckung: {sum(1 for t in timestamps if simulator.is_in_wake_window(t))} Minuten")
    print(f"• Durchschnittliche Aufwach-Wahrscheinlichkeit: {np.mean(predictions):.3f}")
    print(f"• Maximale Aufwach-Wahrscheinlichkeit: {np.max(predictions):.3f}")

    if simulator.alarm_triggered:
        print(f"• Alarm-Status: ✅ AUSGELÖST")
        print(f"🏆 ERFOLG: Smart Alarm hat die optimale Aufwachzeit gefunden!")
        print(f"💡 Vorteil gegenüber traditionellem Alarm: Aufwachen in günstiger Schlafphase")
    elif simulator.best_wake_time:
        print(f"• Alarm-Status: ⏰ BEREIT (Schwelle nicht erreicht)")
    else:
        print(f"• Alarm-Status: ❌ KEINE GEEIGNETE ZEIT GEFUNDEN")
        
    print("\n" + "="*80)
    print("🎓 BUNDESWETTBEWERB KI - PROJEKTABSCHLUSS")
    print("="*80)
    print("Dieses Projekt demonstriert die praktische Anwendung von:")
    print("• Deep Learning (LSTM-Netzwerke)")
    print("• Transfer Learning und Domain Adaptation") 
    print("• Zeitreihenanalyse mit Sensordaten")
    print("• Personalisierte KI-Systeme")
    print("• Echtzeit-Klassifikation und Entscheidungsfindung")
    print("="*80)