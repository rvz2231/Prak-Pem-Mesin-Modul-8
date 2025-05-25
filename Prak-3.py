#1######################################################################################################
# Praktikum 3
# Self-Organizing Map (SOM)

from minisom import MiniSom # [cite: 29]
import numpy as np # [cite: 29]
import matplotlib.pyplot as plt # [cite: 29]
from sklearn import datasets # [cite: 29]

# Gunakan dataset Iris sebagai contoh
# Load dataset
iris = datasets.load_iris() # [cite: 29]
data = iris.data # [cite: 30]

print("Output Point 1: Shape data Iris dan 3 sampel pertama") # Tambahan
print(f"Shape data: {data.shape}")
print(f"Data sampel (3 baris pertama):\n{data[:3]}")

#2######################################################################################################
# Normalisasi data
data = data / data.max(axis=0) # [cite: 31]

print("\nOutput Point 2: Data Iris setelah normalisasi (3 sampel pertama)") # Tambahan
print(f"Data sampel ternormalisasi (3 baris pertama):\n{data[:3]}")

#3######################################################################################################
# Inisialisasi SOM
map_size = (10, 10) # [cite: 31]
# Di PDF: som = MiniSom(map_size[0], map_size[1], data.shape[1], sigma=0.5, learning_rate=0.5)
# Penulisan parameter yang lebih eksplisit (sesuai dokumentasi MiniSom umumnya):
som = MiniSom(x=map_size[0], y=map_size[1], input_len=data.shape[1], #
              sigma=0.5, learning_rate=0.5, random_seed=42) # random_seed tambahan untuk konsistensi

print("\nOutput Point 3: Objek SOM telah diinisialisasi") # Tambahan
print(som)

#4######################################################################################################

# Inisialisasi bobot secara acak
som.random_weights_init(data) # [cite: 32]

print("\nOutput Point 4: Bobot SOM telah diinisialisasi secara acak.") # Tambahan
# Untuk melihat contoh bobot (opsional, bisa sangat banyak):
# print(f"Contoh bobot neuron (0,0):\n{som.get_weights()[0,0,:]}")

#5######################################################################################################
# Pelatihan SOM
num_epochs = 100 # [cite: 33]
som.train_random(data, num_epochs) # [cite: 33]

print("\nOutput Point 5: Pelatihan SOM selesai.") # Tambahan

#6######################################################################################################
# Visualisasi hasil SOM
print("\nOutput Point 6: Tampilan plot hasil SOM (cek window plot)") # Tambahan
plt.figure(figsize=(8, 8)) # [cite: 35]

# Plot U-Matrix
plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=.8) # [cite: 35] alpha disesuaikan
plt.colorbar() # [cite: 35]

# Menandai pemenang untuk setiap sampel
# Di PDF, ada teks `str(i+1)`. Ini akan menandai dengan nomor indeks sampel (1-150).
# Untuk visualisasi yang lebih informatif terkait kelas Iris, kita bisa menggunakan iris.target
# Namun, untuk mengikuti PDF, kita gunakan i+1.
markers = ['o', 's', '^'] # Opsional: penanda berbeda per kelas asli jika diinginkan
colors = ['r', 'g', 'b']  # Opsional: warna berbeda per kelas asli jika diinginkan

# Untuk mengikuti PDF yang hanya menandai nomor sampel:
for i, x_sample in enumerate(data): # [cite: 35]
    w = som.winner(x_sample) # Pemenang untuk sampel x [cite: 35]
    plt.text(w[0] + .5, w[1] + .5, str(i + 1), # [cite: 35]
             color='k', fontdict={'weight': 'bold', 'size': 7}) # size disesuaikan agar tidak terlalu crowded [cite: 35]

# # Alternatif: Menandai dengan label kelas asli (0: setosa, 1: versicolor, 2: virginica)
# for i, x_sample in enumerate(data):
#     w = som.winner(x_sample)
#     plt.text(w[0] + .5, w[1] + .5, str(iris.target[i]),
#              color=colors[iris.target[i]], fontdict={'weight': 'bold', 'size': 11})

plt.title("Self-Organizing Map (SOM) - Iris Dataset") # Tambahan
plt.xticks(np.arange(map_size[0]+1)) # Tambahan: label sumbu x sesuai ukuran peta
plt.yticks(np.arange(map_size[1]+1)) # Tambahan: label sumbu y sesuai ukuran peta
plt.grid(True, linestyle='--', alpha=0.5) # Tambahan: grid untuk kejelasan
plt.show() # [cite: 35]
