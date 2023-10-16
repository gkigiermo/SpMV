# SpMV

The sparse matrix-vector multiplication is one of the most recurrent kernels in scientific simulation.
This repo aims to be a playground to test different storage formats, CUDA architectures, or CUDA functionalities and validate their performance.

The matrix has been generated with a computational fluid dynamics code and represents a Poisson matrix.

Such a matrix is symmetric and positive definitive. Neumann boundary conditions have been considered and the discretization domain was an unstructured mesh using tetrahedron cells.

## Compiling

To compile just use Cmake:

```
mkdir -p build
cmake -S . -B build/
cd build/ ; make
```

## Testing

To run from the build directory execute:

```
./spmv.x ../data/100K
```

## Changing power modes in NVIDIA Xavier AGX

To learn which power mode is activated 
```
sudo /usr/sbin/nvpmodel -q
```

To switch the power modes 
```
sudo /usr/sbin/nvpmodel -m x
```
where x is a number in the range [0,7]
