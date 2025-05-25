#1######################################################################################################
# Persiapan data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('Iris.csv') #
print("Output Point 1: df.head()") # Tambahan untuk menandai output
print(df.head()) #

#2######################################################################################################
# Seleksi Fitur
X = df.iloc[:, 1:-1] # [cite: 8]
y = df.iloc[:, -1] # [cite: 8]
print("\nOutput Point 2: X.head()") # Tambahan untuk menandai output
print(X.head()) # Mirip dengan output di PDF hal. 2 setelah seleksi fitur

#3#####################################################################################################
# Plot Data
# Karena data 4 dimensi, maka akan kita coba
# plot cluster berdasarkan Sepal Length dan Sepal Width saja
print("\nOutput Point 3: Tampilan plot data awal (cek window plot)") # Tambahan
plt.figure(figsize=(8,6)) # Agar plot lebih jelas
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], s=100) # [cite: 10]
plt.title("Data Iris Awal (SepalLength vs SepalWidth)") # Tambahan label
plt.xlabel("SepalLengthCm") # Tambahan label
plt.ylabel("SepalWidthCm") # Tambahan label
plt.show() # Untuk menampilkan plot

#4#####################################################################################################
# Buat Model KMeans
# Kali ini kita coba menggunakan k=2 anggap saja kita tidak tahu jumlah label ada 3 :)

# Inisiasi obyek KMeans
cl_kmeans = KMeans(n_clusters=2) # [cite: 10]
print("\nOutput Point 4: Inisialisasi cl_kmeans") # Tambahan
print(cl_kmeans) # Untuk melihat objek KMeans yang diinisiasi, sesuai PDF

# Fit dan predict model
y_kmeans = cl_kmeans.fit_predict(X) # [cite: 10]
print("\nOutput Point 4: y_kmeans (5 sampel pertama & jumlah cluster)") # Tambahan
print(f"Label cluster untuk 5 sampel pertama: {y_kmeans[:5]}")
print(f"Cluster yang ditemukan: {np.unique(y_kmeans)}")

#5######################################################################################################
# Plot hasil cluster berdasarkan Sepal Length dan Sepal Width
print("\nOutput Point 5: Tampilan plot hasil clustering k=2 (cek window plot)") # Tambahan
plt.figure(figsize=(8,6)) # Agar plot lebih jelas
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], s=100, c=y_kmeans) # [cite: 10]

# Plot centroid
centers = cl_kmeans.cluster_centers_ # [cite: 10]
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5) # [cite: 10]
plt.title("Hasil Clustering K-Means (k=2)") # Tambahan label
plt.xlabel("SepalLengthCm") # Tambahan label
plt.ylabel("SepalWidthCm") # Tambahan label
plt.show() # Untuk menampilkan plot

print("\nOutput Point 5: Koordinat Centroid") # Tambahan
print(centers)

#6######################################################################################################
# Cek Nilai SSE
print("\nOutput Point 6: Nilai SSE untuk k=2") # Tambahan
print(f'Nilai SSE: {cl_kmeans.inertia_}') # [cite: 11]

#7######################################################################################################
# Implementasi Metode Elbow
# List nilai SSE
sse = [] # [cite: 11]
# Cari k terbaik dari 1-10
K = range(1,10) # [cite: 12] (Ini akan menguji k=1 sampai k=9)
                # Untuk k=1 sampai k=10, gunakan range(1,11)

print("\nOutput Point 7: Perhitungan SSE untuk Metode Elbow") # Tambahan
# Cek nilai SSE setiap k
for k_val in K: # [cite: 12]
    kmeanModel = KMeans(n_clusters=k_val) # [cite: 12]
    kmeanModel.fit(X) # [cite: 12]
    sse.append(kmeanModel.inertia_) # [cite: 12]
    print(f"SSE untuk k={k_val}: {kmeanModel.inertia_}") # Tambahan untuk verifikasi langsung

print(f"\nList SSE yang terkumpul: {sse}")

#8######################################################################################################
# Plotting the distortions
print("\nOutput Point 8: Tampilan plot Metode Elbow (cek window plot)") # Tambahan
plt.figure(figsize=(8,4)) # [cite: 12]
plt.plot(K, sse, "bx-") # [cite: 12]
plt.xlabel("k") # [cite: 12]
plt.ylabel("SSE") # [cite: 12]
plt.title("Metode Elbow untuk Mengetahui Jumlah k Terbaik") # [cite: 12]
plt.xticks(list(K)) # Agar semua nilai K terlihat di sumbu-x
plt.grid(True) # Tambahan untuk readability
plt.show() # [cite: 12]

#9######################################################################################################
# Cek Nilai SSE setiap k
print("\nOutput Point 9: Nilai SSE setiap k (sesuai loop PDF)") # Tambahan
# Di PDF: for idx, sse_val in enumerate(sse, start=1):
# Di PDF: print(f'k={idx}; SSE={sse_val}')
# Namun, output yang ditunjukkan di PDF adalah k=0, k=1, dst. yang tidak konsisten dengan start=1
# Saya akan mengikuti kode yang tertulis di PDF halaman 4.
for idx, sse_val in enumerate(sse, start=1): # [cite: 12]
    print(f'k={idx}; SSE={sse_val}') # [cite: 12]