local data_dir = std.extVar("DATA_ROOT") + "/resources/data/tree_dst.agent.data/";
local glove_embeddings = std.extVar("DATA_ROOT") + "/resources/data//glove.840B.300d.zip";

{
  dataset_reader: {
    type: "tree_dst",
    use_agent_utterance: true,
    use_context: true,
    line_limit: 15, 
    source_token_indexers: {
      source_tokens: {
        type: "single_id",
        namespace: "source_tokens",
      },
      source_token_characters: {
        type: "characters",
        namespace: "source_token_characters",
        min_padding_length: 5,
      },
    },
    target_token_indexers: {
      target_tokens: {
        type: "single_id",
        namespace: "target_tokens",
      },
      target_token_characters: {
        type: "characters",
        namespace: "target_token_characters",
        min_padding_length: 5,
      },
    },
    generation_token_indexers: {
      generation_tokens: {
        type: "single_id",
        namespace: "generation_tokens",
      }
    },
    tokenizer: null,
    // tokenizer: {
    //     type: "pretrained_roberta",
    //     model_name: "roberta-base"
    // },
  },
  train_data_path: data_dir + "small",
  validation_data_path: data_dir + "small",
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
      source_tokens: 180000,
      target_tokens: 122000,
      generation_tokens: 122000,
    },
  },
  model: {
    type: "calflow_transformer_parser",
    bert_encoder: null,
    // bert_encoder: {
    //                 type: "seq2seq_roberta_encoder",
    //                 config: "roberta-base",
    //               },
    encoder_token_embedder: {
      token_embedders: {
        source_tokens: {
          type: "embedding",
          vocab_namespace: "source_tokens",
          #pretrained_file: glove_embeddings,
          embedding_dim: 300,
          trainable: true,
        },
        source_token_characters: {
          type: "character_encoding",
          embedding: {
            vocab_namespace: "source_token_characters",
            embedding_dim: 100,
          },
          encoder: {
            type: "cnn",
            embedding_dim: 100,
            num_filters: 50,
            ngram_filter_sizes: [3],
          },
          dropout: 0.33,
        },
      },
    },
    encoder: {
        type: "transformer_encoder",
        input_size: 300 + 50 ,
        hidden_size: 128,
        num_layers: 2,
        encoder_layer: {
            type: "pre_norm",
            d_model: 128,
            n_head: 8,
            norm: {type: "scale_norm",
                  dim: 128},
            dim_feedforward: 512,
            init_scale: 128,  
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
        target_token_characters: {
          type: "character_encoding",
          embedding: {
            vocab_namespace: "target_token_characters",
            embedding_dim: 100,
          },
          encoder: {
            type: "cnn",
            embedding_dim: 100,
            num_filters: 50,
            ngram_filter_sizes: [3],
          },
          dropout: 0.33,
        },
      },
    },
    decoder_node_index_embedding: {
      # vocab_namespace: "node_indices",
      num_embeddings: 500,
      embedding_dim: 50,
    },
    decoder: {
      type: "transformer_decoder",
      input_size: 300 + 50 + 50,
      hidden_size: 128,
      num_layers: 2,
      use_coverage: true,
      decoder_layer: {
        type: "pre_norm",
        d_model: 128, 
        n_head: 8, 
        norm: {type: "scale_norm",
               dim: 128},
        dim_feedforward: 128,
        dropout: 0.2, 
        init_scale: 128,
      },
      source_attention_layer: {
        type: "global",
        query_vector_dim: 128,
        key_vector_dim: 128,
        output_vector_dim: 128,
        attention: {
          type: "mlp",
          # TODO: try to use smaller dims.
          query_vector_dim: 128,
          key_vector_dim: 128,
          hidden_vector_dim: 128, 
          use_coverage: true,
        },
      },
      target_attention_layer: {
        type: "global",
        query_vector_dim: 128,
        key_vector_dim: 128,
        output_vector_dim: 128,
        attention: {
          type: "mlp",
          query_vector_dim: 128,
          key_vector_dim: 128,
          hidden_vector_dim: 128,
          use_coverage: false, 
        },
      },
    },
    extended_pointer_generator: {
      input_vector_dim: 128,
      source_copy: true,
      target_copy: true,
    },
    tree_parser: {
      dropout: 0.20,
      query_vector_dim: 128,
      key_vector_dim: 128,
      edge_head_vector_dim: 128,
      edge_type_vector_dim: 128,
      attention: {
        type: "biaffine",
        query_vector_dim: 128,
        key_vector_dim: 128,
      },
    },
    label_smoothing: {
 	type: "base",
        smoothing: 0.0,
    },
    dropout: 0.2,
    beam_size: 2,
    max_decoding_steps: 100,
    target_output_namespace: "generation_tokens",
    edge_type_namespace: "edge_types",
  },

  iterator: {
    type: "bucket",
    # TODO: try to sort by target tokens.
    sorting_keys: [["source_tokens", "num_tokens"]],
    padding_noise: 0.0,
    batch_size: 85,
  },
  validation_iterator: {
    type: "basic",
    batch_size: 85,
    // instances_per_epoch: 1600,
  },

  trainer: {
    type: "calflow_parsing",
    num_epochs: 2950,
    warmup_epochs: 2940,
    patience: 100000,
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
       model_size: 128, 
       warmup_steps: 8000,
       factor: 1,
    },
    no_grad: [],
    # smatch_tool_path: null, # "smatch_tool",
    validation_data_path: "small",
    validation_prediction_path: "valid_validation.txt",
  },
  random_seed: 12,
  numpy_seed: 12,
  pytorch_seed: 12,
}
