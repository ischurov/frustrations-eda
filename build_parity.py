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

    int calculate_fourier_transform_matrix(
        uint64_t const* states, size_t n_states, uint64_t const* masks, size_t n_masks, int8_t* out
    ) {
        for (size_t i = 0; i < n_states; ++i) {
            for (size_t j = 0; j < n_masks; ++j) {
                out[i * n_masks + j] = __builtin_parityll(states[i] & masks[j]) * 2 - 1;
            }
        }
        return 0;
    }
    """,
    libraries=[],
)

ffibuilder.cdef(
    """
    int parity(uint64_t const* x, uint8_t* out, size_t n);
    int calculate_fourier_transform_matrix(
        uint64_t const* states, size_t n_states, uint64_t const* masks, size_t n_masks, int8_t* out
    );
    """
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
