# Slice with MergeKit

This subfolder contains utilities and scripts that leverage the MergeKit library for slicing (pruning) layers from transformer models as part of an efficient layer-pruning strategy.

MergeKit is a versatile toolkit for model surgery on transformers, allowing for operations such as layer removal and model merging with ease. The provided scripts in this folder are specifically designed to work with the MergeKit API, enabling a seamless layer pruning process.

## Getting Started with MergeKit

Before proceeding with the pruning, ensure you have MergeKit installed. You can set up MergeKit by following these steps:

```bash
git clone https://github.com/cg123/mergekit.git
cd mergekit
pip install -e .
```

After installation, you can use the tools and scripts within this subfolder to prune models based on the computed block similarities as determined by the scripts in the `compute_block_similarity` folder.

## Understanding the Slicing Method

MergeKit's slicing method allows you to selectively remove layers from a pretrained model. This can be a crucial step after identifying which layers or blocks of layers can be pruned without significant impact on model performance.

For detailed documentation on how to use MergeKit for model slicing and the underlying methodology, refer to the official MergeKit repository:

[MergeKit GitHub Repository](https://github.com/arcee-ai/mergekit)

Follow the instructions and examples provided in the MergeKit documentation to understand how to apply the slicing method to your models.

## Next Steps

After pruning your model using MergeKit, you may wish to fine-tune/continual-pretrain the pruned model to restore or even enhance its performance. Fine-tuning should be conducted with a relevant dataset and according to the best practices for training transformer models.
