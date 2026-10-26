# 数据准备

下载并解压后转换为 COCO 目录：`<root>/train2017`、`<root>/val2017` 和 `<root>/annotations/instances_{train,val}2017.json`。

- WTBD: https://doi.org/10.6084/m9.figshare.30210175.v1
- WTBs2025: https://doi.org/10.6084/m9.figshare.28876391.v1

项目不重新分发原始图像或标注。准备完成后运行 `python tools/validate_coco.py <root>`。
