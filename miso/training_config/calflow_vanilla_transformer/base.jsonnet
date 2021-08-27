local data_dir = "/home/t-eliass/resources/data/smcalflow.agent.data/";
local glove_embeddings = "/home/t-eliass/resources/data/glove.840B.300d.zip";

{
  dataset_reader: {
    type: "vanilla_calflow",
    use_agent_utterance: true,
    use_context: true, 
    source_token_indexers: {
      source_tokens: {
        type: "single_id",
        namespace: "source_tokens",
      },
    },
    target_token_indexers: {
      target_tokens: {
        type: "single_id",
        namespace: "target_tokens",
      },
    },
    generation_token_indexers: {
      generation_tokens: {
        type: "single_id",
        namespace: "generation_tokens",
      }
    },
    tokenizer: null,
  },
  train_data_path: data_dir + "train",
  validation_data_path: data_dir + "dev_valid",
  test_data_path: null,
  datasets_for_vocab_creation: [
    "train"
  ],

  vocabulary: {
    non_padded_namespaces: [],
    min_count: {
      source_tokens: 1,
      target_tokens: 1,
      generation_tokens: 1,
    },
    max_vocab_size: {
      source_tokens: 200000,
      target_tokens: 122000,
      generation_tokens: 122000,
    },
  },

  model: {
    type: "vanilla_calflow_transformer_parser",
    fxn_of_interest: "FindManager",
    bert_encoder: null,
    encoder_token_embedder: {
      token_embedders: {
        source_tokens: {
          type: "embedding",
          vocab_namespace: "source_tokens",
          embedding_dim: 300,
          #pretrained_file: glove_embeddings,
          trainable: true,
        },
      },
    },
    encoder: {
        type: "transformer_encoder",
        input_size: 300,
        hidden_size: 512,
        num_layers: 7,
        encoder_layer: {
            type: "pre_norm",
            d_model: 512,
            n_head: 8,
            norm: {type: "scale_norm",
                  dim: 512},
            dim_feedforward: 2048,
            init_scale: 512,  
            },
        dropout: 0.2,
    }, 
    decoder_token_embedder: {
      token_embedders: {
        target_tokens: {
          type: "embedding",
          vocab_namespace: "target_tokens",
          #pretrained_file: glove_embeddings,
          embedding_dim: 300,
          trainable: true,
        },
      },
    },
    decoder: {
      type: "vanilla_transformer_decoder",
      input_size: 300,
      hidden_size: 512,
      num_layers: 4,
      use_coverage: true,
      decoder_layer: {
        type: "pre_norm",
        d_model: 512, 
        n_head: 8, 
        norm: {type: "scale_norm",
               dim: 512},
        dim_feedforward: 2048,
        dropout: 0.2, 
        init_scale: 512,
      },
      source_attention_layer: {
        type: "global",
        query_vector_dim: 512,
        key_vector_dim: 512,
        output_vector_dim: 512,
        attention: {
          type: "mlp",
          # TODO: try to use smaller dims.
          query_vector_dim: 512,
          key_vector_dim: 512,
          hidden_vector_dim: 512, 
          use_coverage: true,
        },
      },
    },
    extended_pointer_generator: {
      input_vector_dim: 512,
      source_copy: true,
    },
    label_smoothing: {
	type: "base",
        smoothing: 0.0,
    },
    dropout: 0.5,
    beam_size: 5,
    max_decoding_steps: 100,
    target_output_namespace: "generation_tokens",
  },

  iterator: {
    type: "bucket",
    # TODO: try to sort by target tokens.
    sorting_keys: [["source_tokens", "num_tokens"]],
    padding_noise: 0.0,
    batch_size: 140,
  },
  validation_iterator: {
    type: "basic",
    batch_size: 140,
    instances_per_epoch: 1600,
  },
  trainer: {
    type: "vanilla_calflow_parsing",
    num_epochs: 250,
    warmup_epochs: 5,
    patience: 20,
    grad_norm: 5.0,
    # TODO: try to use grad clipping.
    grad_clipping: null,
    cuda_device: 0,
    num_serialized_models_to_keep: 1,
    validation_metric: "+exact_match",
    optimizer: {
      type: "adam",
      weight_decay: 3e-9,
      amsgrad: true,
      lr: 0,
    },
    learning_rate_scheduler: {
       type: "noam",
       model_size: 512, 
       warmup_steps: 8000,
       factor: 1,
    },
    no_grad: [],
    # smatch_tool_path: null, # "smatch_tool",
    validation_data_path: "valid",
    validation_prediction_path: "valid_validation.txt",
  },
  random_seed: 12,
  numpy_seed: 12,
  pytorch_seed: 12,
}
