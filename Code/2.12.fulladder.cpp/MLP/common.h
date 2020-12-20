//
// File name: common.h
// Created by ronny on 16-12-2.
//

#ifndef EASYML_COMMON_H
#define EASYML_COMMON_H

namespace easyml {

struct Dim {
    Dim() = default;
    explicit Dim(int b, int c, int h, int w):
        batch_size(b), channels(c), height(h), width(w) {}

    Dim(const Dim &dim) {
        batch_size = dim.batch_size;
        channels = dim.channels;
        height = dim.height;
        width = dim.width;
    }

    Dim &operator=(const Dim &dim) {
        batch_size = dim.batch_size;
        channels = dim.channels;
        height = dim.height;
        width = dim.width;
        return *this;
    }

    int batch_size;
    int channels;
    int height;
    int width;
};

} // namespace easyml

#endif // EASYML_COMMON_H

