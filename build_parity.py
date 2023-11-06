from cffi import FFI

ffibuilder = FFI()

ffibuilder.set_source(
    "_parity",
    r""" 
    #include <assert.h>
    int parity(uint64_t const* x, uint8_t* out, size_t n) {
        assert(sizeof(uint64_t) == sizeof(unsigned long long));
        for (size_t i = 0; i < n; ++i) {
            out[i] = __builtin_parityll(x[i]);
        }
        return 0;
    }

    int popcount(uint64_t const* x, uint8_t* out, size_t n) {
        assert(sizeof(uint64_t) == sizeof(unsigned long long));
        for (size_t i = 0; i < n; ++i) {
            out[i] = __builtin_popcountll(x[i]);
        }
        return 0;
    }

    int calculate_fourier_transform_matrix_int8(
        uint64_t const* states, size_t n_states, uint64_t const* masks, size_t n_masks, int8_t* out
    ) {
        assert(sizeof(uint64_t) == sizeof(unsigned long long));
        
        #pragma omp parallel for
        for (size_t i = 0; i < n_states; ++i) {
            for (size_t j = 0; j < n_masks; ++j) {
                out[i * n_masks + j] = (1 - __builtin_parityll(states[i] & masks[j]) * 2);
            }
        }
        return 0;
    }

    int calculate_fourier_transform_matrix_float64(
        uint64_t const* states, size_t n_states, uint64_t const* masks, size_t n_masks, double* out
    ) {
        assert(sizeof(uint64_t) == sizeof(unsigned long long));
        
        #pragma omp parallel for
        for (size_t i = 0; i < n_states; ++i) {
            for (size_t j = 0; j < n_masks; ++j) {
                out[i * n_masks + j] = (double)((1 - __builtin_parityll(states[i] & masks[j]) * 2));
            }
        }
        return 0;
    }
    """,
    libraries=[],
    extra_compile_args=["-O3", "-march=nehalem", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)

ffibuilder.cdef(
    """
    int parity(uint64_t const* x, uint8_t* out, size_t n);
    int popcount(uint64_t const* x, uint8_t* out, size_t n);
    int calculate_fourier_transform_matrix_int8(
        uint64_t const* states, size_t n_states, uint64_t const* masks, size_t n_masks, int8_t* out
    );
    int calculate_fourier_transform_matrix_float64(
        uint64_t const* states, size_t n_states, uint64_t const* masks, size_t n_masks, double* out
    );
    
    """
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
