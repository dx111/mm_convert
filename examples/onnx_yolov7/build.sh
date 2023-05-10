g++ infer.cpp \
    -o infer.bin \
    -I/usr/local/neuware/include \
    -L/usr/local/neuware/lib64 \
    -lopencv_core \
    -lopencv_imgcodecs \
    -lopencv_imgproc \
    -lmagicmind_runtime \
    -lcnrt