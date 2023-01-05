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
    """,
    libraries=[],
)

ffibuilder.cdef(
    """
    int parity(uint64_t const* x, uint8_t* out, size_t n);
    """
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
