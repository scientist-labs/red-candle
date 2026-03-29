# frozen_string_literal: true

require_relative "smoke_test_helper"

section "Tokenizer"

tok = test("load") { Candle::Tokenizer.from_pretrained("bert-base-uncased") }

if tok
  test("encode") { tok.encode("Hello, world!") }
  test("encode (no special tokens)") { tok.encode("Hello", add_special_tokens: false) }
  test("encode_to_tokens") { tok.encode_to_tokens("Hello, world!") }
  test("encode_with_tokens") do
    result = tok.encode_with_tokens("Hello")
    raise "missing ids" unless result[:ids]
    raise "missing tokens" unless result[:tokens]
    result
  end
  test("encode_batch") { tok.encode_batch(["Hello", "World"]) }
  test("decode") { tok.decode([101, 7592, 102]) }
  test("get_vocab") { tok.get_vocab }
  test("vocab_size") do
    size = tok.vocab_size
    raise "unexpected size" unless size > 1000
    size
  end
  test("id_to_token") { tok.id_to_token(101) }
  test("with_padding") { tok.with_padding(length: 128) }
  test("with_truncation") { tok.with_truncation(512) }
end

smoke_test_summary
