from torch.utils.data import DataLoader
from peanut.model_train.data_loader import MoleculeDataset



# dataset = MoleculeDataset("data/a_wpS.pt")
# loader = DataLoader(dataset, batch_size=32, shuffle=True)


# ---------------------------------------
# Beispielnutzung
# ---------------------------------------
if __name__ == "__main__":
    dataset_path = "data/a_wpS.pt"  # dein gespeichertes Dataset
    dataset = MoleculeDataset(dataset_path)

    # Länge des Datasets abfragen
    print(f"Dataset enthält {len(dataset)} Moleküle.")

    # Ein einzelnes Molekül abrufen (z.B. das erste)
    mol = dataset[0]
    print("\nErstes Molekül:")
    for key, value in mol.items():
        print(f"{key}: {type(value)}, shape: {getattr(value, 'shape', None)}")

    # Ein zufälliges Molekül abrufen
    import random
    idx = random.randint(0, len(dataset) - 1)
    mol = dataset[idx]
    print(f"\nZufälliges Molekül (Index {idx}): Energie = \n{mol['energy']:.4f} eV")
    print(f"\nZufälliges Molekül (Index {idx}): Kräfte = \n{mol["forces"][:3]} eV")
