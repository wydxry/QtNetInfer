TEMPLATE = lib
CONFIG += staticlib
CONFIG -= app_bundle
CONFIG -= qt

# 具体实现封装成自定义静态库
TARGET = Net
DEFINES += NET_LIBRARY

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

INCLUDEPATH += "E:\libtorch-win-shared-with-deps-2.0.1+cu118\libtorch\include"
INCLUDEPATH += "E:\libtorch-win-shared-with-deps-2.0.1+cu118\libtorch\include\torch\csrc\api\include"
LIBS += -LE:/libtorch-win-shared-with-deps-2.0.1+cu118/libtorch/lib \
        -lasmjit \
        -lc10 \
        -lc10_cuda \
        # -lcaffe2_detectron_ops_gpu \
        # -lcaffe2_module_test_dynamic \
        -lcaffe2_nvrtc \
        # -lCaffe2_perfkernels_avx \
        # -lCaffe2_perfkernels_avx2 \
        # -lCaffe2_perfkernels_avx512 \
        -lclog \
        -lcpuinfo \
        -ldnnl \
        -lfbgemm \
        -lfbjni \
        -lkineto \
        -llibprotobuf \
        -llibprotobuf-lite \
        -llibprotoc \
        # -lmkldnn \
        -lpthreadpool \
        -lpytorch_jni \
        -ltorch \
        -ltorch_cpu \
        -ltorch_cuda \
        # -ltorch_cuda_cpp \
        # -ltorch_cuda_cu \
        -lXNNPACK

LIBS += -INCLUDE:"?ignore_this_library_placeholder@@YAHXZ"

INCLUDEPATH += "E:\codes\QtCodes\QtNetInferWithLib\include"

HEADERS += \
    include/net.h

SOURCES += \
        src/net.cpp \

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target


INCLUDEPATH += "E:\OpenCV\build\include" \
               # "E:\OpenCV\build\include\opencv" \
               "E:\OpenCV\build\include\opencv2"


# 添加v14版 opencv 库文件，区分debug和release
win32:CONFIG(release, debug|release): LIBS += -LE:\OpenCV\build\x64\vc16\lib -lopencv_world470
else:win32:CONFIG(debug, debug|release): LIBS += -LE:\OpenCV\build\x64\vc16\lib -lopencv_world470d

INCLUDEPATH += E:\OpenCV\build\include\opencv2
DEPENDPATH += E:\OpenCV\build\include\opencv2

