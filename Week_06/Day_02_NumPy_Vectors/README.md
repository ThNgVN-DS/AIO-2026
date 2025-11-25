# 📊 Week 06, Day 02 - NumPy: Vectors & Distance Metrics

**Course:** AI Việt Nam All-in-One 2026  
**Instructor:** Dr. Quang-Vinh Dinh  
**Date:** November 24, 2025  
**Status:** ✅ Completed

---

## 📚 Learning Objectives

This session provides a comprehensive foundation in NumPy and vector mathematics essential for AI/ML development:

### Part I: NumPy Fundamentals
- ✅ Understanding `ndarray` architecture (shape, dtype, ndim)
- ✅ 7+ methods for array creation
- ✅ Indexing & Slicing (View vs Copy mechanics)
- ✅ Broadcasting rules for efficient computation
- ✅ Universal Functions (ufuncs) for vectorization
- ✅ Axis operations and dimensional manipulation

### Part II: Vector Mathematics
- ✅ Vector operations (addition, scalar multiplication, dot product)
- ✅ Vector norms (L1, L2, L-infinity)
- ✅ Distance metrics (Euclidean, Manhattan, Cosine)
- ✅ Unit vectors and normalization

### Part III: Practical Applications
- ✅ K-Nearest Neighbors (KNN) implementation from scratch
- ✅ Image processing (RGB to Grayscale conversion)
- ✅ IoU & NMS for object detection
- ✅ Set operations with arrays

### Part IV: Best Practices
- ✅ Performance optimization techniques
- ✅ Common pitfalls (view vs copy, broadcasting errors)
- ✅ Professional coding standards
- ✅ 10 MCQ assessment questions

---

## 📁 Repository Contents

```
Day_02_NumPy_Vectors/
├── README.md                                    # This file
├── W06_D02_NumPy_Vector_Distance_Complete_Guide.ipynb   # Main notebook
└── assets/                                      # Supporting materials
```

---

## 🔑 Key Concepts

### 1. NumPy Performance Advantage

**Benchmark Results:**
- Python List: 1.544 seconds
- NumPy Array: 0.040 seconds
- **Performance Gain: 38.2x faster** ⚡

**Why NumPy is Faster:**
- Contiguous memory layout
- Vectorized operations in C/Fortran
- SIMD (Single Instruction Multiple Data)
- Optimized BLAS/LAPACK backends

### 2. Broadcasting Rules

NumPy automatically expands dimensions for element-wise operations:

```python
# Example: Adding 1D array to 2D matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])  # Shape: (2, 3)
vector = np.array([10, 20, 30])  # Shape: (3,)

result = matrix + vector  # Broadcasting!
# Result:
# [[11, 22, 33],
#  [14, 25, 36]]
```

### 3. Distance Metrics Comparison

| Metric | Formula | Use Case | Sensitivity |
|--------|---------|----------|-------------|
| **Euclidean** | √Σ(aᵢ-bᵢ)² | General distance, KNN | High to outliers |
| **Manhattan** | Σ\|aᵢ-bᵢ\| | Grid-based, city distance | Robust to outliers |
| **Cosine** | 1 - (a·b)/(\|a\|\|b\|) | Text similarity, angles | Magnitude-independent |

### 4. View vs Copy (Critical!)

```python
# View (shares memory)
slice_view = arr[1:4]  # Changes affect original

# Copy (independent)
slice_copy = arr[1:4].copy()  # Changes are isolated
```

---

## 💡 Practical Implementation Highlights

### KNN Algorithm from Scratch

```python
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            # Compute distances to all training points
            distances = np.linalg.norm(self.X_train - x, axis=1)
            
            # Get K nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            
            # Vote (classification)
            prediction = np.bincount(k_labels).argmax()
            predictions.append(prediction)
        
        return np.array(predictions)
```

---

## 🎯 Assessment Performance

### MCQ Topics Covered
1. ✅ Broadcasting mechanics
2. ✅ View vs Copy behavior
3. ✅ Axis operations
4. ✅ Dot product calculations
5. ✅ Reshape rules
6. ✅ Boolean indexing
7. ✅ Cosine similarity interpretation
8. ✅ Auto-dimension calculation (-1)
9. ✅ Copy methods
10. ✅ KNN edge cases (K=1)

**Score:** Comprehensive understanding demonstrated ✅

---

## 📖 References & Further Reading

### Core Textbooks
- **Mathematics for Machine Learning** (Deisenroth et al., 2020)
  - Chapter 2: Linear Algebra
  - Chapter 3: Analytic Geometry

- **Hands-On Machine Learning** (Géron, 2023)
  - Chapter 4: Training Linear Models
  - Chapter 10: Neural Networks with NumPy

### Online Resources
- [NumPy Official Documentation](https://numpy.org/doc/)
- [Mathematics for ML Companion Website](https://mml-book.github.io/)
- Coursera: Mathematics for Machine Learning Specialization

### Performance Optimization
- Intel MKL (Math Kernel Library) integration
- BLAS/LAPACK backend configuration
- SIMD instruction utilization (AVX2)

---

## 🏷️ Tags

`numpy` `linear-algebra` `vectors` `distance-metrics` `knn` `machine-learning` `python` `data-science` `aio-2026` `broadcasting` `vectorization`

---

## 📊 Learning Progress

- [x] NumPy fundamentals mastered
- [x] Vector mathematics applied
- [x] Distance metrics implemented
- [x] KNN algorithm built from scratch
- [x] Best practices internalized
- [x] Assessment completed

**Next Steps:**
- Week 06, Day 03: Advanced Linear Algebra
- Matrix decompositions (SVD, QR, Cholesky)
- Eigenvalues & Eigenvectors
- PCA implementation

---

## 👨‍🎓 Student Notes

**Key Takeaways:**
1. **Vectorize Everything** - Avoid Python loops when NumPy operations are available
2. **Think in Shapes** - Always be aware of array dimensions for broadcasting
3. **View vs Copy Awareness** - Slicing creates views; use `.copy()` for independence
4. **Performance Matters** - NumPy is 10-100x faster than pure Python

**Personal Insights:**
- Broadcasting initially confusing but extremely powerful once understood
- KNN implementation deepened understanding of distance metrics
- View/Copy distinction critical for debugging

---

**Course Repository:** [AIO-2026](https://github.com/ThNgVN-DS/AIO-2026)  
**Instructor:** Dr. Quang-Vinh Dinh  
**Institution:** AI Việt Nam

---

*Last Updated: November 25, 2025*  
*Status: Ready for Portfolio Review* ✅
