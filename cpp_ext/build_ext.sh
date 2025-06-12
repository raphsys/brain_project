#!/bin/bash

set -e  # arrêt sur erreur

echo "=================================="
echo " PYTORCH EXTENSION COMPILATION "
echo "=================================="

# Compilation C++ Extension
python3 setup.py install --user

echo "✅ Compilation terminée."

# Résolution dynamique des dépendances (libc10.so)
TORCH_LIB_PATH=$(python3 -c "import torch; import os; print(os.path.join(torch.__path__[0],'lib'))")

echo "Ajout automatique de Torch lib à LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=${TORCH_LIB_PATH}:$LD_LIBRARY_PATH

echo "LD_LIBRARY_PATH configuré automatiquement."
echo "Tu peux maintenant lancer Python et faire : import memory_ext"

# Petit test immédiat pour vérifier l'import fonctionne
python3 -c "import memory_ext; print('✅ memory_ext importé avec succès')"

