local data_dir = "/mnt/default/resources/data/smcalflow_samples/FindManager/5000_100/";
local glove_embeddings = "/mnt/default/resources/data/glove.840B.300d.zip";

{
  dataset_reader: {
    type: "calflow",
    use_agent_utterance: true,
    use_context: true,
    #line_limit: 2, 
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
    tokenizer: {
        type: "pretrained_transformer_for_amr",
        args: null,
        kwargs: {
            do_lowercase: false
        },
        model_name: "bert-base-cased"
    },
  },
  train_data_path: data_dir + "train",
  validation_data_path: data_dir + "valid",
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
      source_tokens: 18000,
      target_tokens: 5000,
      generation_tokens: 500,
    },
  },

  model: {
    type: "calflow_parser",
    bert_encoder: {
        "type": "seq2seq_bert_encoder",
        "config": "bert-base-cased"
    },
    encoder_token_embedder: {
      token_embedders: {
        source_tokens: {
          type: "embedding",
          vocab_namespace: "source_tokens",
          embedding_dim: 300,
          pretrained_file: glove_embeddings,
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
      type: "miso_stacked_bilstm",
      batch_first: true,
      stateful: true,
      input_size: 300 + 50 + 768,
      hidden_size: 512,
      num_layers: 2,
      recurrent_dropout_probability: 0.33,
      use_highway: false,
    },
    decoder_token_embedder: {
      token_embedders: {
        target_tokens: {
          type: "embedding",
          vocab_namespace: "target_tokens",
          pretrained_file: glove_embeddings,
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
      rnn_cell: {
        input_size: 300 + 50 + 50 + 256 + 768,
        hidden_size: 1024,
        num_layers: 2,
        recurrent_dropout_probability: 0.33,
        use_highway: false,
      },
      source_attention_layer: {
        type: "global",
        query_vector_dim: 1024,
        key_vector_dim: 1024,
        output_vector_dim: 1024,
        attention: {
          type: "mlp",
          # TODO: try to use smaller dims.
          query_vector_dim: 1024,
          key_vector_dim: 1024,
          hidden_vector_dim: 256, 
          use_coverage: true,
        },
      },
      target_attention_layer: {
        type: "global",
        query_vector_dim: 1024,
        key_vector_dim: 1024,
        output_vector_dim: 1024,
        attention: {
          type: "mlp",
          query_vector_dim: 1024,
          key_vector_dim: 1024,
          hidden_vector_dim: 256,
          use_coverage: false,
        },
      },
      dropout: 0.33,
    },
    extended_pointer_generator: {
      input_vector_dim: 1024,
      source_copy: true,
      target_copy: true,
    },
    tree_parser: {
      query_vector_dim: 1024,
      key_vector_dim: 1024,
      edge_head_vector_dim: 256,
      edge_type_vector_dim: 128,
      attention: {
        type: "biaffine",
        query_vector_dim: 256,
        key_vector_dim: 256,
      },
      dropout: 0.33
    },
    label_smoothing: {
        smoothing: 0.0,
    },
    dropout: 0.00,
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
    batch_size: 300,
  },
  validation_iterator: {
    type: "basic",
    batch_size: 300,
    instances_per_epoch: 1600,
  },

  trainer: {
    type: "calflow_parsing",
    num_epochs: 250,
    warmup_epochs: 0,
    patience: 250,
    grad_norm: 5.0,
    # TODO: try to use grad clipping.
    grad_clipping: null,
    cuda_device: 0,
    num_serialized_models_to_keep: 2,
    validation_metric: "+exact_match",
    optimizer: {
      type: "adam",
      weight_decay: 3e-9,
      amsgrad: true,
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
