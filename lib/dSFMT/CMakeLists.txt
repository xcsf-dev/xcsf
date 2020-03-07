set(dSFMT_SOURCES
    dSFMT.c
)

set(dSFMT_HEADERS
    dSFMT-common.h
    dSFMT-params.h
    dSFMT-params19937.h
    dSFMT.h
)

add_library(dSFMT STATIC ${dSFMT_SOURCES} ${dSFMT_HEADERS})
