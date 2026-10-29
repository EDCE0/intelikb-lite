# 知识库目录

把待检索的文档放在这里，支持：

- `.pdf`（按页解析，保留页码）
- `.txt` / `.md` / `.markdown`
- `.docx`

放好后在项目根目录执行：

```bash
python scripts/build_index.py      # 重建索引
python scripts/ask.py -q "你的问题"  # 命令行提问
```

> 说明：仓库不包含原始文献 PDF（多为期刊论文，涉及版权且体积较大），
> 请自行准备资料放入本目录。扫描版 PDF 需先做 OCR，否则无法抽取文本。
