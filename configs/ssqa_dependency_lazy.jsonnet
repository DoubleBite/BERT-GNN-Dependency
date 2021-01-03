local transformer_model = 'bert-base-chinese';
local epochs = 20;
local batch_size = 8;

{
  "dataset_reader": {
      "type": "ssqa_dependency",
      "transformer_model_name": transformer_model,
      "skip_invalid_examples": true,
      "lazy": true
      //"max_instances": 200  // debug setting
  },
  "validation_dataset_reader": self.dataset_reader + {
      "skip_invalid_examples": false,
  },
  "train_data_path": "data/ssqa_multiple_choice_with_dependency/train.json",
  "validation_data_path": "data/ssqa_multiple_choice_with_dependency/dev.json",
  "model": {
      "type": "ssqa_gnn",
      "transformer_model_name": transformer_model,
      "gnn_encoder":{
        "type": "gcn",
        "input_dim": 768, # The output dim for bert embedders
        "hidden_dim":500,
        "output_dim": 300,
      },
  },
  "data_loader": {
    "batch_size": batch_size,
  },
  "trainer": {
    "optimizer": {
      "type": "huggingface_adamw",
      "weight_decay": 0.0,
      "parameter_groups": [[["bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0}]],
      "lr": 2e-5,
      "eps": 1e-8,
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": epochs,
      "cut_frac": 0.1,
    },
    "grad_clipping": 1.0,
    "num_epochs": epochs,
    "validation_metric": "+per_instance_f1",
    // "cuda_device":-1,
    "tensorboard_writer":{
      "histogram_interval":10
    },
  },
//  "random_seed": 42,
//  "numpy_seed": 42,
//  "pytorch_seed": 42,
}