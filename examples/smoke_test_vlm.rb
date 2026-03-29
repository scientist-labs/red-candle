# frozen_string_literal: true

require_relative "smoke_test_helper"

section "Vision-Language Model"

vlm = test("load (LLaVA-Next Vicuna 7B)") do
  Candle::VLM.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf", device: $device)
end

if vlm
  test_image = File.join(__dir__, "test_cat.jpg")
  if File.exist?(test_image)
    test("ask") do
      result = vlm.ask(test_image, "What animal is in this image?", max_length: 50)
      raise "empty response" if result.strip.empty?
      result
    end
    test("describe") do
      result = vlm.describe(test_image, max_length: 50)
      raise "empty response" if result.strip.empty?
      result
    end
  else
    puts "  Skipping image tests (examples/test_cat.jpg not found)"
  end
  test("model_id") { vlm.model_id }
  test("tokenizer access") { vlm.tokenizer }
  test("inspect") { vlm.inspect }

  vlm = nil
  GC.start(full_mark: true, immediate_sweep: true)
end

smoke_test_summary
