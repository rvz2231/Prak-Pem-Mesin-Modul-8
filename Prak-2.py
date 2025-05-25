#1A######################################################################################################
# Praktikum 2
# Konsep K-Means untuk klasterisasi data
import matplotlib.pyplot as plt #
import seaborn as sns; sns.set() #
import numpy as np #
from sklearn.datasets import make_blobs #

# Membuat data blobs
X_blobs, y_true_blobs = make_blobs(n_samples=300, centers=4, #
cluster_std=0.60, random_state=0) #
# Plot data blobs awal
print("Output Point 1A: Tampilan plot data blobs awal (cek window plot)") # Tambahan
plt.figure(figsize=(8, 6)) # Tambahan
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], s=50) #
plt.title("Data Blobs Sintetik Awal") # Tambahan
plt.xlabel("Fitur 1") # Tambahan
plt.ylabel("Fitur 2") # Tambahan
plt.show() #

#1B######################################################################################################
from sklearn.cluster import KMeans

# Inisialisasi dan fit model KMeans
# Di PDF: kmeans = KMeans(n_clusters=4)
# Di PDF: kmeans.fit(X_blobs)
# Di PDF: y_kmeans_blobs = kmeans.predict(X_blobs)
# Kita bisa gabungkan fit dan predict
kmeans_blobs = KMeans(n_clusters=4, random_state=0, n_init='auto') 
# random_state untuk konsistensi, n_init untuk warning
y_kmeans_blobs = kmeans_blobs.fit_predict(X_blobs) #

# Plot hasil clustering
print("\nOutput Point 1B: Tampilan plot data blobs ter-cluster (cek window plot)") # Tambahan
plt.figure(figsize=(8, 6)) # Tambahan
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_kmeans_blobs, s=50, cmap='viridis') #

centers_blobs = kmeans_blobs.cluster_centers_ #
plt.scatter(centers_blobs[:, 0], centers_blobs[:, 1], c='black', s=200, alpha=0.5) #
plt.title("Hasil Clustering K-Means pada Data Blobs") # Tambahan
plt.xlabel("Fitur 1") # Tambahan
plt.ylabel("Fitur 2") # Tambahan
plt.show() #

#2A######################################################################################################
from sklearn.metrics import pairwise_distances_argmin #

def find_clusters(X, n_clusters, rseed=2): #
# 1. Randomly choose clusters
    rng = np.random.RandomState(rseed) #
    i = rng.permutation(X.shape[0])[:n_clusters] #
    centers = X[i] #

    while True: #
        # 2a. Assign labels based on closest center (E-step)
        labels = pairwise_distances_argmin(X, centers) #

        # 2b. Find new centers from means of points (M-step)
        new_centers = np.array([X[labels == i].mean(0) #
                                for i in range(n_clusters)]) #

        # 2c. Check for convergence
        if np.all(centers == new_centers): #
            break #
        centers = new_centers #

    return centers, labels #

print("\nOutput Point 2A: Fungsi find_clusters telah didefinisikan.") # Tambahan

#2B######################################################################################################

# Menjalankan find_clusters pada X_blobs (data dari Point 1)
# Sesuai PDF menggunakan rseed=2 by default
centers_manual_2, labels_manual_2 = find_clusters(X_blobs, 4, rseed=2) 

# Plot hasil clustering dari find_clusters
print("\nOutput Point 2B: Tampilan plot hasil clustering find_clusters (rseed=2) (cek window plot)") # Tambahan
plt.figure(figsize=(8, 6)) # Tambahan
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels_manual_2, s=50, cmap='viridis') #
plt.scatter(centers_manual_2[:, 0], centers_manual_2[:, 1], c='black', s=200, alpha=0.75) # Tambahan centroid
plt.title("Hasil Clustering Manual E-M (find_clusters, rseed=2)") # Tambahan
plt.xlabel("Fitur 1") # Tambahan
plt.ylabel("Fitur 2") # Tambahan
plt.show() #

#3######################################################################################################

# Perubahan random (menggunakan rseed=0)
centers_manual_0, labels_manual_0 = find_clusters(X_blobs, 4, rseed=0) #

# Plot hasil clustering dari find_clusters dengan rseed=0
print("\nOutput Point 3: Tampilan plot hasil clustering find_clusters (rseed=0) (cek window plot)") # Tambahan
plt.figure(figsize=(8, 6)) # Tambahan
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels_manual_0, s=50, cmap='viridis'); # Titik koma di PDF di akhir baris ini
plt.scatter(centers_manual_0[:, 0], centers_manual_0[:, 1], c='black', s=200, alpha=0.75) # Tambahan centroid
plt.title("Hasil Clustering Manual E-M (find_clusters, rseed=0)") # Tambahan
plt.xlabel("Fitur 1") # Tambahan
plt.ylabel("Fitur 2") # Tambahan
plt.show() #

#4######################################################################################################

# Optimalisasi Jumlah Klaster (Contoh menggunakan k=6 pada data dengan 4 pusat)
# Di PDF: labels = KMeans(6, random_state=0).fit_predict(X_blobs)
# Di PDF: plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels, s=50, cmap='viridis');
kmeans_6 = KMeans(n_clusters=6, random_state=0, n_init='auto') #
labels_6_blobs = kmeans_6.fit_predict(X_blobs) #

print("\nOutput Point 4: Tampilan plot clustering k=6 pada data blobs (cek window plot)") # Tambahan
plt.figure(figsize=(8, 6)) # Tambahan
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels_6_blobs, s=50, cmap='viridis') 
plt.scatter(kmeans_6.cluster_centers_[:,0], kmeans_6.cluster_centers_[:,1], c='black', s=100, alpha=0.5) # Tambahan centroid
plt.title("Hasil Clustering K-Means (k=6) pada Data Blobs (4 Pusat Asli)") # Tambahan
plt.xlabel("Fitur 1") # Tambahan
plt.ylabel("Fitur 2") # Tambahan
plt.show() #

#5######################################################################################################
from sklearn.datasets import make_moons #

# Membuat data moons
X_moons, y_moons_true = make_moons(200, noise=.05, random_state=0) #

# Menerapkan KMeans pada data moons
kmeans_moons = KMeans(n_clusters=2, random_state=0, n_init='auto') #
labels_moons_kmeans = kmeans_moons.fit_predict(X_moons) #

# Plot hasil KMeans pada data moons
print("\nOutput Point 5: Tampilan plot K-Means pada data moons (cek window plot)") # Tambahan
plt.figure(figsize=(8, 6)) # Tambahan
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_moons_kmeans, s=50, cmap='viridis') # Sesuai PDF 'C=labels'
plt.scatter(kmeans_moons.cluster_centers_[:,0], kmeans_moons.cluster_centers_[:,1], c='black', s=200, alpha=0.5) # Tambahan centroid
plt.title("Hasil K-Means pada Data Moons") # Tambahan
plt.xlabel("Fitur 1") # Tambahan
plt.ylabel("Fitur 2") # Tambahan
plt.show() #

#6######################################################################################################
from sklearn.cluster import SpectralClustering #

# Menerapkan Spectral Clustering pada data moons
model_spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', #
                                 assign_labels='kmeans', random_state=0) # random_state untuk konsistensi
labels_moons_spectral = model_spectral.fit_predict(X_moons) #

# Plot hasil Spectral Clustering pada data moons
print("\nOutput Point 6: Tampilan plot Spectral Clustering pada data moons (cek window plot)") # Tambahan
plt.figure(figsize=(8, 6)) # Tambahan
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_moons_spectral, s=50, cmap='viridis') # Sesuai PDF 'C=labels'
plt.title("Hasil Spectral Clustering pada Data Moons") # Tambahan
plt.xlabel("Fitur 1") # Tambahan
plt.ylabel("Fitur 2") # Tambahan
plt.show() #


#7######################################################################################################
from sklearn.datasets import load_sample_image #

# Memuat gambar contoh
flower = load_sample_image("flower.jpg") #

# Menampilkan gambar asli
print("\nOutput Point 7: Tampilan citra asli 'flower.jpg' (cek window plot) dan shape-nya") # Tambahan
plt.figure(figsize=(8,6)) # Tambahan
ax = plt.axes(xticks=[], yticks=[]) #
ax.imshow(flower) #
plt.title("Citra Asli 'flower.jpg'") # Tambahan
plt.show() #

print(f"Shape citra asli: {flower.shape}") #

#8######################################################################################################
# Normalisasi data gambar (nilai piksel menjadi 0-1)
data_flower = flower / 255.0 #
# Reshape data gambar menjadi array 2D (jumlah_piksel x 3 channel RGB)
# Di PDF: data = data.reshape(427 * 640, 3)
# Sebaiknya gunakan shape dari variabel flower agar dinamis
data_flower_reshaped = data_flower.reshape(flower.shape[0] * flower.shape[1], flower.shape[2]) #

print("\nOutput Point 8: Shape data citra setelah di-reshape") # Tambahan
print(f"Shape data setelah reshape: {data_flower_reshaped.shape}") #


#9######################################################################################################

def plot_pixels(data, title, colors=None, N=10000): #
    if colors is None: #
        colors = data #
    # choose a random subset
    rng = np.random.RandomState(0) #
    i = rng.permutation(data.shape[0])[:N] #
    colors_subset = colors[i] # Variabel colors di PDF di-overwrite, sebaiknya bedakan
    data_subset = data[i] # Variabel data di PDF di-overwrite, sebaiknya bedakan

    R, G, B = data_subset.T #

    fig, ax = plt.subplots(1, 2, figsize=(16, 6)) #
    ax[0].scatter(R, G, color=colors_subset, marker='.') #
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1)) #

    ax[1].scatter(R, B, color=colors_subset, marker='.') #
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1)) #

    fig.suptitle(title, size=20); #

print("\nOutput Point 9A: Fungsi plot_pixels telah didefinisikan.") # Tambahan

print("\nOutput Point 9B: Tampilan plot ruang warna asli (cek window plot)") # Tambahan
plot_pixels(data_flower_reshaped, title='Input color space: 16 million possible colors') #
plt.show() # Tambahan untuk memastikan plot muncul


#10######################################################################################################
import warnings; warnings.simplefilter('ignore') # Fix NumPy issues. (Sesuai PDF)

from sklearn.cluster import MiniBatchKMeans #

# Inisialisasi dan fit MiniBatchKMeans untuk menemukan 16 warna utama
kmeans_image = MiniBatchKMeans(n_clusters=16, random_state=0, n_init='auto') # random_state untuk konsistensi
kmeans_image.fit(data_flower_reshaped) #

# Dapatkan warna baru berdasarkan prediksi cluster (centroid)
new_colors = kmeans_image.cluster_centers_[kmeans_image.predict(data_flower_reshaped)] #

print("\nOutput Point 10: Proses reduksi warna selesai. `new_colors` telah dibuat.") # Tambahan
print(f"Shape `new_colors`: {new_colors.shape}")
print(f"Contoh 5 warna baru pertama:\n{new_colors[:5]}")

#11######################################################################################################
print("\nOutput Point 11: Tampilan plot ruang warna hasil reduksi (cek window plot)") # Tambahan
plot_pixels(data_flower_reshaped, colors=new_colors, #
            title="Reduced color space: 16 colors") #
plt.show() # Tambahan untuk memastikan plot muncul

#12######################################################################################################
# Reshape new_colors ke bentuk gambar asli
flower_recolored = new_colors.reshape(flower.shape) #

# Menampilkan gambar asli dan gambar hasil reduksi warna
print("\nOutput Point 12: Tampilan perbandingan citra asli dan hasil reduksi (cek window plot)") # Tambahan
fig, ax = plt.subplots(1, 2, figsize=(16, 6), #
                       subplot_kw=dict(xticks=[], yticks=[])) #
fig.subplots_adjust(wspace=0.05) #
ax[0].imshow(flower) #
ax[0].set_title('Original Image', size=16) #

ax[1].imshow(flower_recolored) #
ax[1].set_title('16-color Image', size=16); #

plt.show() # Tambahan untuk memastikan plot muncul
