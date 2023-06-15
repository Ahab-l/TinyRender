# TinyRender
基于c++ 语言的软光栅
# 原开源项目地址
https://github.com/ssloy/tinyrenderer
# 对源项目的改进或修改
 源项目中使用的透视矩阵比起传统的“压缩”视锥体远平面更像是“扩张视锥体”近平面，后续很多算法实现与此相关，而作者也为足够详细涉及其原理。我选择继续使用我认为更传统正确的投影矩阵，并基于此在完成了后续的开发。后续涉及到切线空间的纹理映射，以及shadowmap等。同时也对源项目的一些错误的渲染过程做出了自己的矫正。


# 渲染结果
![Image](https://github.com/Ahab-l/render-image/blob/main/tinyrender/ambient.png)
![Image](https://github.com/Ahab-l/render-image/blob/main/tinyrender/framebuffer.png)
![Image}(https://github.com/Ahab-l/render-image/blob/main/tinyrender/goroudshading.png)

