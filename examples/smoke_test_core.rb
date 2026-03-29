# frozen_string_literal: true

require_relative "smoke_test_helper"

section "Build Info"
info = Candle::BuildInfo.summary
puts "  Device: #{info[:default_device]} | Metal: #{info[:metal_available]} | CUDA: #{info[:cuda_available]}"

section "GenerationConfig"

test("deterministic preset") { Candle::GenerationConfig.deterministic(max_length: 10) }
test("creative preset") { Candle::GenerationConfig.creative(max_length: 10) }
test("balanced preset") { Candle::GenerationConfig.balanced(max_length: 10) }
test(".with override") do
  config = Candle::GenerationConfig.balanced(max_length: 10)
  config2 = config.with(temperature: 0.5)
  raise "temperature not updated" unless config2.temperature == 0.5
  raise "max_length not preserved" unless config2.max_length == 10
  config2
end
test("inspect") { Candle::GenerationConfig.balanced.inspect }

section "Tensor"

test("Tensor.new") { Candle::Tensor.new([1.0, 2.0, 3.0]) }
test("Tensor.zeros") { Candle::Tensor.zeros([2, 3]) }
test("Tensor.ones") { Candle::Tensor.ones([2, 3]) }
test("Tensor.rand") { Candle::Tensor.rand([2, 3]) }
test("to_f / to_i") do
  t = Candle::Tensor.new([42.0]).squeeze(0)
  raise "to_f failed" unless t.to_f == 42.0
  raise "to_i failed" unless t.to_i == 42
  t
end
test("each / Enumerable") do
  t = Candle::Tensor.new([1.0, 2.0, 3.0])
  values = t.map { |x| x }
  raise "wrong count" unless values.length == 3
  values
end

smoke_test_summary
