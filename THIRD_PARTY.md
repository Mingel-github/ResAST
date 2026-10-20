# Third party sources

## MeowAgeNet

Data, author embeddings and author-derived head recipes originate from:

van Toor, A., Qazi, N. and Paladini, S. (2025). *A deep learning pipeline for age prediction from vocalisations of the domestic feline*. Scientific Reports 15, 34565. https://doi.org/10.1038/s41598-025-17986-z

Source repository: https://github.com/aster-droide/feline-age-prediction

Pinned revision: `3d02295bef1500d2b2500a124596f77010181391`

The upstream work declares Creative Commons Attribution 4.0 International. Its license text is retained in `third_party/meowagenet-CC-BY-4.0.txt`. Derived metadata adds duplicate/identity cleaning and fixed role assignments; local paths have been made portable. Audio and source embedding CSVs are obtained separately from upstream. The source provides its work without warranties under its license.

## Pretrained models and implementations

| Component | Source |
| --- | --- |
| AST | https://github.com/YuanGongND/ast |
| Transformers AST implementation | https://github.com/huggingface/transformers/tree/v4.48.3/src/transformers/models/audio_spectrogram_transformer |
| PANNs CNN14 | https://github.com/qiuqiangkong/audioset_tagging_cnn |
| PANNs inference package | https://github.com/qiuqiangkong/panns_inference |
| AVES | https://github.com/earthspecies/aves |
| SSAST | https://github.com/YuanGongND/ssast |
| SSAST runtime conversion | https://huggingface.co/Simon-Kotchou/ssast-base-patch-audioset-16-16 |
| librosa acoustic analysis | https://github.com/librosa/librosa |

AST and SSAST use the Hugging Face implementations. AVES and PANNs use their installed inference packages. Framework code and pretrained weights are dependencies, not re-licensed copies included in this repository. Model IDs, revisions and downloaded-file hashes are recorded in `configs/pretrained.json`. Consult each source for its own usage conditions.

The multiscale attention wrapper and acoustic residual are implemented in `src/resast/models/`. Equal probability averaging is an established ensemble technique; its implementation here combines independently initialized training runs.

Original project code is licensed under MIT; see [LICENSE](LICENSE). The MeowAgeNet-derived metadata and author-derived materials retain the upstream CC BY 4.0 terms. Dependencies and downloaded model weights retain their respective licenses.
