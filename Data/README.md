---
base_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:3725
- loss:CosineSimilarityLoss
widget:
- source_sentence: GMさんもサプライチェーンの再構築を図りながらインセンティブが得られる状 況に持っていこうと努力しており、この点にも期待しながら来期に臨んでいく。
  sentences:
  - 'Question 6: Is the increase in inventory due to the impact of logistics?'
  - We improved profitability for about 60% of customers through the measures.
  - GM is making efforts to restructure the supply chain to become eligible for incentives,
    and we are hopeful about this move as we approach the next fiscal year.
- source_sentence: その上で、逆イールド カーブが継続する前提に立ち、米国債のベア型ファンドや、米金利スワップの固定払 い・変動受け取引等のショートポジションを起点に、機動的なポジション量の伸縮で売買
    益を積み上げ、所有期間利回り全体の向上を図る。
  sentences:
  - There has been a substantial decline in engagement with PS4 third-party titles,
    especially previous catalog titles.
  - With this approach, based on the premise that the inverted yield curve will continue,
    we will increase capital gains through flexible changes to our positions and aim
    to raise the overall return of the holding period, starting with U.S. Treasury
    Bond bear funds and short positions such as fixed payment and variable receipt
    U.S. interest swap transactions.
  - 'Question 5: Retail sales in the European market have increased compared to the
    previous period, but the operating profit in Europe is in the red.'
- source_sentence: 横田 [A]：大体の構成費として、うちの売り上げに占める韓国の割合って、通常だと大体4割ぐら いが通常の構成費なんですけれども、韓国のポリシーチェンジによる流通在庫調整が入って、Q１
    に関しては約50%対前年で影響しているというのが一番の大きな影響です。
  sentences:
  - '[A]:For the South Korea ratio of our company sales, about 40% is usually the
    portion.'
  - For JCIB, RWA is converted to capital, and ROE is calculated based on that capital.
  - Therefore, the majority of this comes from the delay in recovery of Chinese travelers
    to South Korea than expected.
- source_sentence: ② アライアンスにおいて、ソフトウェアの開発費の削減のイメージを教えてほしい。
  sentences:
  - 2. How you envision the reduction of software development costs in the alliance?
  - Please refer to page three.
  - We will consider the further acceleration of the divestment of equity holdings
    in fiscal 2022.
- source_sentence: ACURAと Hondaに分けると、Hondaはしっかりと抑えながらやれている。
  sentences:
  - We will also reallocate RWAs to support initiatives to resolve issues requiring
    a cross-sector approach.
  - If we divide the situation into Acura and Honda, we can say that Honda is doing
    a good job while keeping a firm grip on the situation.
  - Investment Management newly established in Aprilplaysa key roleon our strategy
    to expand into private markets and this quarter a private equity investee company
    listed and we saw a strong contributiontoearnings.
---

# SentenceTransformer based on sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) <!-- at revision bf3bf13ab40c3157080a7ab344c831b9ad18b5eb -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 384 tokens
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'ACURAと Hondaに分けると、Hondaはしっかりと抑えながらやれている。',
    'If we divide the situation into Acura and Honda, we can say that Honda is doing a good job while keeping a firm grip on the situation.',
    'Investment Management newly established in Aprilplaysa key roleon our strategy to expand into private markets and this quarter a private equity investee company listed and we saw a strong contributiontoearnings.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset


* Size: 3,725 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         | label                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | float                                                         |
  | details | <ul><li>min: 5 tokens</li><li>mean: 36.92 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 32.75 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 1.0</li><li>mean: 1.0</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                | sentence_1                                                                                                                                                                                                                   | label            |
  |:------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>これに加えて、アプリで完結できる手続きや、タブレットでの取引を拡充することで、コー ルセンターや店頭での事務量を削減し、これを経費削減に繋げていきたい。</code> | <code>In addition, we want to reduce the administration volume at call centers and branch counters by enhancing procedures that can be completed using apps and increasing transactions with tablets to reduce costs.</code> | <code>1.0</code> |
  | <code>次に9ページ、その他の地域事業についてです。</code>                                                       | <code>Next, on page nine, I would like to discuss our other regional businesses.</code>                                                                                                                                      | <code>1.0</code> |
  | <code>何となくもっと利 益が出るかなと思っていましたので、確認させてください。</code>                                         | <code>I was wondering if it would be more profitable for some reason, so let me confirm on that.</code>                                                                                                                      | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 2
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `eval_use_gather_object`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Framework Versions
- Python: 3.10.12
- Sentence Transformers: 3.1.1
- Transformers: 4.44.2
- PyTorch: 2.4.1+cu121
- Accelerate: 0.34.2
- Datasets: 3.0.1
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->