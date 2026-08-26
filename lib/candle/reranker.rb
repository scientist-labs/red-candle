module Candle
  class Reranker
    # Default model path for cross-encoder/ms-marco-MiniLM-L-12-v2
    DEFAULT_MODEL_PATH = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    # Default number of query-document pairs processed per forward pass. Keeps peak
    # tensor size bounded on CPU regardless of how many candidates are passed to #rerank.
    DEFAULT_BATCH_SIZE = 16

    # Load a pre-trained reranker model from HuggingFace
    # @param model_id [String] HuggingFace model ID (defaults to cross-encoder/ms-marco-MiniLM-L-12-v2)
    # @param device [Candle::Device] The device to use for computation (defaults to best available)
    # @param max_length [Integer] Maximum sequence length for truncation (defaults to 512)
    # @param revision [String, nil] Immutable HF revision (commit SHA, tag, or branch) to pin the
    #   model/tokenizer/config download to. When given, files are fetched from exactly that
    #   revision and never fall back to the repository's default branch. When omitted, the
    #   repository's default branch ("main") is tracked, matching prior behavior.
    # @param batch_size [Integer] Number of query-document pairs processed per forward pass
    #   (defaults to 16). Large document arrays are split into micro-batches instead of building
    #   one tensor for every candidate; must be a positive integer.
    # @return [Reranker] A new Reranker instance
    def self.from_pretrained(model_id = DEFAULT_MODEL_PATH, device: Candle::Device.best, max_length: 512,
                              revision: nil, batch_size: DEFAULT_BATCH_SIZE)
      validate_batch_size!(batch_size)
      _create(model_id, device, max_length, revision, batch_size)
    end

    # Constructor for creating a new Reranker with optional parameters
    # @deprecated Use {.from_pretrained} instead
    # @param model_path [String, nil] The path to the model on Hugging Face
    # @param device [Candle::Device, Candle::Device.cpu] The device to use for computation
    # @param max_length [Integer] Maximum sequence length for truncation (defaults to 512)
    # @param revision [String, nil] Immutable HF revision to pin the download to
    # @param batch_size [Integer] Micro-batch size for reranking (defaults to 16)
    def self.new(model_path: DEFAULT_MODEL_PATH, device: Candle::Device.best, max_length: 512,
                 revision: nil, batch_size: DEFAULT_BATCH_SIZE)
      $stderr.puts "[DEPRECATION] `Reranker.new` is deprecated. Please use `Reranker.from_pretrained` instead."
      validate_batch_size!(batch_size)
      _create(model_path, device, max_length, revision, batch_size)
    end

    # @param batch_size [Object] candidate batch_size value to validate
    # @raise [ArgumentError] if batch_size is not a positive Integer
    def self.validate_batch_size!(batch_size)
      unless batch_size.is_a?(Integer) && batch_size >= 1
        raise ArgumentError, "batch_size must be a positive integer, got #{batch_size.inspect}"
      end
    end
    private_class_method :validate_batch_size!

    # Returns documents ranked by relevance using the specified pooling method.
    # @param query [String] The input text
    # @param documents [Array<String>] The list of documents to compare against
    # @param pooling_method [String] Pooling method: "pooler", "cls", or "mean". Default: "pooler"
    # @param apply_sigmoid [Boolean] Whether to apply sigmoid to the scores. Default: true
    def rerank(query, documents, pooling_method: "pooler", apply_sigmoid: true)
      rerank_with_options(query, documents, pooling_method, apply_sigmoid).collect { |doc, score, doc_id|
        { doc_id: doc_id, score: score, text: doc }
      }
    end
    
    # Improved inspect method
    def inspect
      opts = options rescue {}
      parts = ["#<Candle::Reranker"]
      parts << "model=#{opts["model_id"] || "unknown"}"
      parts << "revision=#{opts["revision"] || "main"}"
      parts << "device=#{opts["device"] || "unknown"}"
      parts << "batch_size=#{opts["batch_size"] || "unknown"}"
      parts.join(" ") + ">"
    end
  end
end
