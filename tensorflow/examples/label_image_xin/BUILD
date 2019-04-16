# Description:
#   TensorFlow C++ inference example for labeling images.

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

exports_files(["data/V_R194_T1_MICX0.000_MICZ0.000.jpg"])

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "siemens",
    srcs = [
        "main.cc",
    ],
    linkstatic = 1,
    linkshared = 0,
    linkopts = select({
        "//tensorflow:android": [
            "-pie",
            "-landroid",
            "-ljnigraphics",
            "-llog",
            "-lm",
            "-z defs",
            "-s",
            "-Wl,--exclude-libs,ALL",
        ],
        "//conditions:default": ["-lm"],
    }),
    deps = select({
        "//tensorflow:android": [
            # cc:cc_ops is used to include image ops (for siemens)
            # Jpg, gif, and png related code won't be included
            "//tensorflow/cc:cc_ops",
            "//tensorflow/core:android_tensorflow_lib",
            # cc:android_tensorflow_image_op is for including jpeg/gif/png
            # decoder to enable real-image evaluation on Android
            "//tensorflow/core/kernels:android_tensorflow_image_op",
        ],
        "//conditions:default": [
            "//tensorflow/cc:cc_ops",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:tensorflow",
        ],
    }),
)

py_binary(
    name = "siemens_py",
    srcs = ["siemens.py"],
    main = "siemens.py",
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)
