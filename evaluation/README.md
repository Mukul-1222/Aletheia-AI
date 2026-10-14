# Evaluation strategy

Aletheia uses an independent fixed reference corpus for general language quality. Before a serious run, build it with:

```powershell
python -m aletheia --build-eval-corpus
```

Guided learning in FINAL9.5 also evaluates each update against two additional tracks when available:

- the current lesson's held-out chunks;
- held-out chunks from previously validated lessons.

The fixed reference is kept separate from training. The other two tracks are held-out portions of validated developmental experiences and are used only as secondary transfer/stability evidence.

Do not put the fixed evaluation file inside the teaching path. For a foundation run, build the fixed reference before `--pretrain` and exclude the raw training directory through `runtime.evaluation_exclude_paths` when overlap is possible.
